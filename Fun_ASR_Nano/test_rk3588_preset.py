#!/usr/bin/env python3
"""Test script to verify RK3588 preset configuration."""

from Optimize_ONNX import get_device_preset

print("=" * 70)
print("RK3588 Preset Configuration Test")
print("=" * 70)

preset = get_device_preset('rk3588')

print(f"\nDevice: RK3588")
print(f"Description: {preset['description']}")
print("\nConfiguration:")
print(f"  • Quantization Type: {preset['quant'].upper()}")
print(f"  • Bits: {preset['bits']}")
print(f"  • Algorithm: {preset['algorithm']}")
print(f"  • Block Size: {preset['block_size']}")
print(f"  • Opset Version: {preset['opset']}")
print(f"  • Symmetric: {preset['symmetric']}")
print(f"  • Accuracy Level: {preset['accuracy_level']}")
print(f"  • Hybrid Mode: {preset['rk3588_hybrid']}")
print(f"  • Encoder INT8: {preset['rk3588_encoder_int8']}")

print("\nOptimization Notes:")
for note in preset['notes']:
    print(f"  {note}")

print("\n" + "=" * 70)
print("✅ Verification Complete!")
print("=" * 70)

# Verify INT4 is NOT used
assert preset['quant'] != 'int4', "ERROR: RK3588 preset should not use INT4!"
assert preset['bits'] == 8, "ERROR: RK3588 preset should use 8 bits!"
print("\n✅ All assertions passed - RK3588 correctly configured for INT8 (not INT4)")
