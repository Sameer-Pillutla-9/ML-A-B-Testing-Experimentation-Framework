from ab_test import frequentist_z_test, bayesian_prob_beats_control


if __name__ == "__main__":
    # Example: Test B has a slightly higher price with similar conversion
    success_a, total_a = 120, 1000
    success_b, total_b = 135, 1000

    res = frequentist_z_test(success_a, total_a, success_b, total_b)
    prob = bayesian_prob_beats_control(success_a, total_a, success_b, total_b)

    print(f"A conversion: {res.variant_a_rate:.3f}")
    print(f"B conversion: {res.variant_b_rate:.3f}")
    print(f"Lift: {res.lift * 100:.2f}%")
    print(f"z = {res.z_score:.3f}, p = {res.p_value:.4f}")
    print(f"P(B > A) ≈ {prob:.3f}")
