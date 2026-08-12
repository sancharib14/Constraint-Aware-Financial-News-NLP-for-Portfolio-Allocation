import pandas as pd

def _cap_and_allocate_side(
    desired_abs_weights,
    target_exposure: float,
    max_abs_weight: float
):
    """
    Allocate one side of the book under a max weight constraint.

    desired_abs_weights:
        positive desired weights for one side, e.g. long side.

    target_exposure:
        total exposure for this side, e.g. 0.5.

    max_abs_weight:
        max allowed absolute weight per name.

    Logic:
        Start from desired relative weights.
        Cap names above max_abs_weight.
        Redistribute leftover exposure to uncapped names.
        Repeat until feasible.
    """

    desired = desired_abs_weights.copy().astype(float)

    if desired.sum() <= 0:
        return desired * 0.0

    # Feasibility check
    max_possible = len(desired) * max_abs_weight
    actual_target = min(target_exposure, max_possible)

    weights = desired / desired.sum() * actual_target

    capped = weights * 0.0
    remaining_names = weights.index.tolist()
    remaining_exposure = actual_target

    while len(remaining_names) > 0:
        sub_desired = desired.loc[remaining_names]

        if sub_desired.sum() <= 0:
            alloc = pd.Series(
                remaining_exposure / len(remaining_names),
                index=remaining_names
            )
        else:
            alloc = sub_desired / sub_desired.sum() * remaining_exposure

        too_big = alloc > max_abs_weight

        if not too_big.any():
            capped.loc[remaining_names] = alloc
            break

        big_names = alloc[too_big].index

        capped.loc[big_names] = max_abs_weight

        remaining_exposure -= max_abs_weight * len(big_names)
        remaining_names = [name for name in remaining_names if name not in big_names]

        if remaining_exposure <= 1e-12:
            break

    return capped


def clip_and_renormalize_weights(
    weights: pd.DataFrame,
    max_abs_weight: float = 0.10,
    target_gross_exposure: float = 1.0,
    date_col: str = "date",
    ticker_col: str = "ticker",
    weight_col: str = "weight"
) -> pd.DataFrame:
    """
    Strictly repair portfolio weights.

    This version enforces:
        abs(weight_i) <= max_abs_weight

    It repairs long and short sides separately.

    Example:
        target_gross_exposure = 1.0
        long side target = +0.5
        short side target = -0.5
    """

    repaired_list = []

    side_exposure = target_gross_exposure / 2.0

    for date, group in weights.groupby(date_col):
        g = group.copy()

        longs = g[g[weight_col] > 0].copy()
        shorts = g[g[weight_col] < 0].copy()

        repaired_parts = []

        if len(longs) > 0:
            desired_long_abs = longs.set_index(ticker_col)[weight_col].abs()

            long_alloc = _cap_and_allocate_side(
                desired_abs_weights=desired_long_abs,
                target_exposure=side_exposure,
                max_abs_weight=max_abs_weight
            )

            long_df = long_alloc.reset_index()
            long_df.columns = [ticker_col, weight_col]
            long_df[date_col] = date
            repaired_parts.append(long_df)

        if len(shorts) > 0:
            desired_short_abs = shorts.set_index(ticker_col)[weight_col].abs()

            short_alloc = _cap_and_allocate_side(
                desired_abs_weights=desired_short_abs,
                target_exposure=side_exposure,
                max_abs_weight=max_abs_weight
            )

            short_df = short_alloc.reset_index()
            short_df.columns = [ticker_col, weight_col]
            short_df[weight_col] = -short_df[weight_col]
            short_df[date_col] = date
            repaired_parts.append(short_df)

        if len(repaired_parts) > 0:
            repaired_day = pd.concat(repaired_parts, ignore_index=True)
            repaired_list.append(repaired_day)

    if len(repaired_list) == 0:
        return weights.copy()

    repaired = pd.concat(repaired_list, ignore_index=True)

    repaired = repaired[[date_col, ticker_col, weight_col]]
    repaired = repaired.sort_values([date_col, ticker_col]).reset_index(drop=True)

    return repaired


def check_weight_constraints(
    weights: pd.DataFrame,
    max_abs_weight: float = 0.10,
    date_col: str = "date",
    ticker_col: str = "ticker",
    weight_col: str = "weight"
) -> pd.DataFrame:
    """
    Check simple single-name weight constraint.

    A violation happens when abs(weight) > max_abs_weight.
    """

    df = weights.copy()

    df["abs_weight"] = df[weight_col].abs()
    df["violates_max_abs_weight"] = df["abs_weight"] > max_abs_weight

    summary = (
        df.groupby(date_col)
        .agg(
            num_positions=(ticker_col, "count"),
            max_abs_weight=("abs_weight", "max"),
            num_weight_violations=("violates_max_abs_weight", "sum"),
        )
        .reset_index()
    )

    summary["has_weight_violation"] = summary["num_weight_violations"] > 0

    return summary




def apply_turnover_limit(
    weights: pd.DataFrame,
    max_turnover: float = 0.50,
    date_col: str = "date",
    ticker_col: str = "ticker",
    weight_col: str = "weight"
) -> pd.DataFrame:
    """
    Smooth portfolio weights to limit daily turnover.

    If today's desired weights are too different from yesterday's weights,
    blend between yesterday and today's desired weights.

    This directly reduces transaction cost and unstable trading.
    """

    wide = (
        weights.pivot_table(
            index=date_col,
            columns=ticker_col,
            values=weight_col,
            aggfunc="sum"
        )
        .fillna(0.0)
        .sort_index()
    )

    repaired_rows = []
    prev = None

    for date, desired in wide.iterrows():
        desired = desired.copy()

        if prev is None:
            final = desired
        else:
            diff = desired - prev
            turnover = diff.abs().sum()

            if turnover <= max_turnover or turnover == 0:
                final = desired
            else:
                # alpha controls how much of the desired trade we allow
                alpha = max_turnover / turnover
                final = prev + alpha * diff

        final_df = final.reset_index()
        final_df.columns = [ticker_col, weight_col]
        final_df[date_col] = date

        repaired_rows.append(final_df)
        prev = final

    repaired = pd.concat(repaired_rows, ignore_index=True)

    # Remove tiny zero-weight rows to keep dataframe clean
    repaired = repaired[repaired[weight_col].abs() > 1e-12].copy()

    repaired = repaired[[date_col, ticker_col, weight_col]]
    repaired = repaired.sort_values([date_col, ticker_col]).reset_index(drop=True)

    return repaired


def summarize_constraint_violations(
    constraint_df: pd.DataFrame,
    date_col: str = "date"
) -> dict:
    """
    Summarize constraint violations across time.
    """

    if len(constraint_df) == 0:
        return {
            "violation_days": 0,
            "violation_rate": 0.0,
            "avg_num_weight_violations": 0.0,
            "avg_max_abs_weight": 0.0,
        }

    return {
        "violation_days": int(constraint_df["has_weight_violation"].sum()),
        "violation_rate": float(constraint_df["has_weight_violation"].mean()),
        "avg_num_weight_violations": float(constraint_df["num_weight_violations"].mean()),
        "avg_max_abs_weight": float(constraint_df["max_abs_weight"].mean()),
    }
