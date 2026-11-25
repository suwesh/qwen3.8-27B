import argparse

p = argparse.ArgumentParser()
p.add_argument("--weight-gb", type=float, required=True)
p.add_argument("--bandwidth-gbs", type=float, required=True)
p.add_argument("--measured-tps", type=float, required=True)
a = p.parse_args()

roof = a.bandwidth_gbs / a.weight_gb
effective = a.measured_tps * a.weight_gb
eff = 100.0 * a.measured_tps / roof

print(f"Weight bytes per token: {a.weight_gb:.6f} GB")
print(f"Bandwidth ceiling:      {a.bandwidth_gbs:.3f} GB/s")
print(f"Decode roofline:        {roof:.3f} tok/s")
print(f"Measured decode:        {a.measured_tps:.3f} tok/s")
print(f"Effective bandwidth:    {effective:.3f} GB/s")
print(f"Roofline efficiency:    {eff:.2f}%")
