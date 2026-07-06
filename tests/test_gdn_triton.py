"""Numerical validation: Triton GDN kernel vs Python reference.

Run on a CUDA machine:
    python -m pytest tests/test_gdn_triton.py -v
or directly:
    python tests/test_gdn_triton.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pytest
import torch
import torch.nn.functional as F

from transformer_lm.model.gdn_triton import gdn_triton_available, gdn_triton_forward, _MAX_S_ELEMS
from transformer_lm.model.gated_delta_net import _python_scan


def _make_inputs(B, H, N, d_k, d_v, device, dtype=torch.float32, seed=42):
    torch.manual_seed(seed)
    Q = torch.randn(B, H, N, d_k, device=device, dtype=dtype)
    K = torch.randn(B, H, N, d_k, device=device, dtype=dtype)
    V = torch.randn(B, H, N, d_v, device=device, dtype=dtype)
    gamma = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype))
    beta  = torch.sigmoid(torch.randn(B, H, N, device=device, dtype=dtype))
    phi_Q = F.elu(Q) + 1.0
    phi_K = F.elu(K) + 1.0
    return phi_Q, phi_K, V, gamma, beta


# ── CPU / reference tests (always run) ────────────────────────────────────────

def test_python_scan_output_shape():
    phi_Q, phi_K, V, gamma, beta = _make_inputs(2, 4, 16, 64, 64, "cpu")
    out = _python_scan(phi_Q, phi_K, V, gamma, beta, eps=1e-6)
    assert out.shape == (2, 4, 16, 64)


def test_python_scan_no_nan():
    phi_Q, phi_K, V, gamma, beta = _make_inputs(1, 2, 64, 64, 64, "cpu")
    out = _python_scan(phi_Q, phi_K, V, gamma, beta, eps=1e-6)
    assert torch.isfinite(out).all(), "NaN/Inf in Python scan output"


def test_python_scan_causal():
    """Changing future tokens must not affect past outputs."""
    B, H, N, d = 1, 2, 32, 64
    phi_Q, phi_K, V, gamma, beta = _make_inputs(B, H, N, d, d, "cpu")
    out_ref = _python_scan(phi_Q, phi_K, V, gamma, beta, eps=1e-6)

    # Perturb the second half of V, K, Q, gamma, beta
    phi_Q2, phi_K2, V2, gamma2, beta2 = (
        phi_Q.clone(), phi_K.clone(), V.clone(), gamma.clone(), beta.clone()
    )
    phi_Q2[:, :, N // 2:] = torch.randn_like(phi_Q2[:, :, N // 2:])
    phi_K2[:, :, N // 2:] = torch.randn_like(phi_K2[:, :, N // 2:])
    V2[:, :,    N // 2:] = torch.randn_like(V2[:, :, N // 2:])
    gamma2[:, :, N // 2:] = torch.rand_like(gamma2[:, :, N // 2:])
    beta2[:, :,  N // 2:] = torch.rand_like(beta2[:, :, N // 2:])

    out_perturbed = _python_scan(phi_Q2, phi_K2, V2, gamma2, beta2, eps=1e-6)
    # First half must be identical
    torch.testing.assert_close(out_ref[:, :, :N // 2], out_perturbed[:, :, :N // 2])


# ── Triton tests (CUDA only) ───────────────────────────────────────────────────

@pytest.mark.skipif(not gdn_triton_available(), reason="Triton + CUDA not available")
@pytest.mark.parametrize("B,H,N,d", [
    (1, 1,  16, 64),    # tiny
    (2, 4,  64, 64),    # typical small model
    (1, 16, 128, 64),   # typical medium model
    (2, 4, 512, 64),    # longer sequence
])
def test_triton_matches_python(B, H, N, d):
    device = "cuda"
    phi_Q, phi_K, V, gamma, beta = _make_inputs(B, H, N, d, d, device)

    ref = _python_scan(phi_Q, phi_K, V, gamma, beta, eps=1e-6)
    tri = gdn_triton_forward(
        phi_Q.contiguous(), phi_K.contiguous(), V.contiguous(),
        gamma.contiguous(), beta.contiguous(), eps=1e-6,
    )

    torch.testing.assert_close(ref, tri, atol=1e-4, rtol=1e-3,
                                msg=f"Mismatch for B={B} H={H} N={N} d={d}")


@pytest.mark.skipif(not gdn_triton_available(), reason="Triton + CUDA not available")
def test_triton_non_power_of_2_dk():
    """xl preset has d_k=80 — verify padding/masking is correct."""
    B, H, N, d_k, d_v = 1, 4, 32, 80, 80
    phi_Q, phi_K, V, gamma, beta = _make_inputs(B, H, N, d_k, d_v, "cuda")

    ref = _python_scan(phi_Q, phi_K, V, gamma, beta, eps=1e-6)
    tri = gdn_triton_forward(
        phi_Q.contiguous(), phi_K.contiguous(), V.contiguous(),
        gamma.contiguous(), beta.contiguous(), eps=1e-6,
    )

    torch.testing.assert_close(ref, tri, atol=1e-4, rtol=1e-3,
                                msg="Mismatch for xl-preset (d_k=80, non-power-of-2)")


@pytest.mark.skipif(not gdn_triton_available(), reason="Triton + CUDA not available")
def test_triton_fallback_for_large_state():
    """When d_k * d_v > _MAX_S_ELEMS, causal_gated_delta_net must use Python scan."""
    from transformer_lm.model.gated_delta_net import causal_gated_delta_net

    # Force a size that exceeds budget
    d = 200   # 200*200 = 40000 > 16384
    assert d * d > _MAX_S_ELEMS, "Test assumption violated: adjust d"

    # Just check it doesn't crash and produces finite values
    B, H, N = 1, 1, 8
    Q = torch.randn(B, H, N, d, device="cuda")
    K = torch.randn(B, H, N, d, device="cuda")
    V = torch.randn(B, H, N, d, device="cuda")
    gamma = torch.sigmoid(torch.randn(B, H, N, device="cuda"))
    beta  = torch.sigmoid(torch.randn(B, H, N, device="cuda"))

    out = causal_gated_delta_net(Q, K, V, gamma, beta)
    assert out.shape == (B, H, N, d)
    assert torch.isfinite(out).all()


if __name__ == "__main__":
    print("Python scan shape test ...", end=" ")
    test_python_scan_output_shape(); print("OK")

    print("Python scan no-NaN test ...", end=" ")
    test_python_scan_no_nan(); print("OK")

    print("Python scan causal test ...", end=" ")
    test_python_scan_causal(); print("OK")

    if gdn_triton_available():
        print("Triton vs Python (small) ...", end=" ")
        test_triton_matches_python(2, 4, 64, 64); print("OK")

        print("Triton vs Python (medium) ...", end=" ")
        test_triton_matches_python(1, 16, 128, 64); print("OK")

        print("Triton non-power-of-2 d_k ...", end=" ")
        test_triton_non_power_of_2_dk(); print("OK")

        print("Triton fallback for large state ...", end=" ")
        test_triton_fallback_for_large_state(); print("OK")
    else:
        print("Triton/CUDA not available — skipping GPU tests")

    print("All tests passed.")
