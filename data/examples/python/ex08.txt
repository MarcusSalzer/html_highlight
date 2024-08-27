pred_prob_test = model(X_test_norm)

result = pl.DataFrame(
    {
        "id": test_ids,
        "Response": pred_prob_test,
    }
)

result.write_parquet("result.parquet")