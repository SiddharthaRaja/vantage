"""Prices that come from ${VAR} expansion arrive as strings and must still compute."""

from vantage.backends.base import Pricing


def test_string_prices_are_coerced():
    p = Pricing(kind="self_hosted", gpu_hourly_usd="0.6", n_gpus="2")
    assert p.gpu_hourly_usd == 0.6 and p.n_gpus == 2


def test_unset_or_zero_gpu_price_means_unpriced():
    assert Pricing(gpu_hourly_usd="").gpu_hourly_usd is None
    assert Pricing(gpu_hourly_usd="0.0").gpu_hourly_usd is None


def test_api_prices_coerced():
    p = Pricing(kind="api", input_usd_per_mtok="0.15", output_usd_per_mtok="0.6")
    assert (p.input_usd_per_mtok, p.output_usd_per_mtok) == (0.15, 0.6)
