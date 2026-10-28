"""Offline synthetic illustrations of loss and first-default probability algebra."""

from creditratingsystem import (
    build_calculation_context,
    cumulative_pd_from_conditional,
    expected_loss,
    marginal_pd_from_conditional,
)


def main():
    """Use made-up, deterministic inputs; no files or services are needed."""
    print("Synthetic algebraic examples; amounts use the same monetary unit.")
    context = build_calculation_context(data_manifest={
        "data_origin": "SYNTHETIC", "parameter_origin": "SYNTHETIC_REFERENCE",
        "dataset_id": "bootstrap_handcrafted",
    })
    loss = expected_loss(pd=0.02, lgd=0.4, ead=1000, context=context)
    print("Expected default loss:", loss.value)
    print("Raw economic LGD examples:",
          expected_loss([0.2], [-0.25, 1.5], [100], context=context).value)

    hazards = [0.1, 0.2]
    cumulative = cumulative_pd_from_conditional(hazards, context=context)
    marginal = marginal_pd_from_conditional(hazards, context=context)
    print("Cumulative first-default PD:", cumulative.value)
    print("Marginal first-default PD:", marginal.value)
    print("Method:", cumulative.method_id)
    print("Unit:", cumulative.unit)

    absorbed = marginal_pd_from_conditional([0.25, 1.0, 0.9], context=context)
    print("First-default masses with absorption:", absorbed.value)
    print("Tiny hazards:",
          cumulative_pd_from_conditional([1e-20, 2e-20], context=context).value)


if __name__ == "__main__":
    main()
