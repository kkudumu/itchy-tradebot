# Expected value per trade in R-multiples
# E[R] = WR * RR - (1 - WR)

win_rates = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60]
rr_ratios = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

print("Expected Value per Trade (R-multiples)")
print("=" * 90)
header = f"{'WR':>6} |" + "|".join(f" {rr:.1f}:1  " for rr in rr_ratios)
print(header)
print("-" * 90)

for wr in win_rates:
    row = f"{wr*100:.0f}%   |"
    for rr in rr_ratios:
        ev = wr * rr - (1 - wr)
        row += f" {ev:+.3f} |"
    print(row)

print()
print("Breakeven Win Rate for each RR:")
for rr in rr_ratios:
    be_wr = 1 / (1 + rr)
    print(f"  RR {rr:.1f}:1 -> Breakeven WR = {be_wr*100:.1f}%")
