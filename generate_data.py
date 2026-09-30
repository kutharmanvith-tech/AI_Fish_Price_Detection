from datetime import date
from pathlib import Path

from forecasting import make_demo_data


if __name__ == "__main__":
    output = Path(__file__).parent / "data" / "fish_prices.csv"
    output.parent.mkdir(exist_ok=True)
    start = date(date.today().year - 5, date.today().month, date.today().day).isoformat()
    make_demo_data(start=start).to_csv(output, index=False)
    print(f"Created {output}")