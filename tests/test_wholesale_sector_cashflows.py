"""Economic rights and independent sector cash/capital checks."""
from pathlib import Path
import runpy


def test_wholesale_sector_cashflows():
    result=runpy.run_path(str(Path(__file__).parents[1]/'examples/wholesale_sector_cashflows.py'))['run']()
    rows={r['variant']:r for r in result['cases']}
    assert rows['financial_holding']['independent_cash_oracle'][0]['financial_credit_unpaid']==3
    assert rows['financial_holding']['balance_stress']['restricted_capital_not_cash']==200
    assert [rows[v]['balance_stress']['stressed_capital']for v in ['life','nonlife','reinsurer']]==[10,105,98]
    assert [rows[v]['independent_cash_oracle'][1]['financial_credit_unpaid']for v in ['region','municipality','public_agency']]==[25,0,30]
    assert rows['public_agency']['independent_cash_oracle'][1]['operating_unpaid']==45
    assert rows['reinsurer']['independent_cash_oracle'][0]['counterparty_receipts']==50
    assert rows['reinsurer']['independent_cash_oracle'][1]['counterparty_receipts']==0
    for v,amount in [('multilateral_lender',67),('supranational_body',35)]:
        flows=rows[v]['receivables']['contract']['cashflows']
        assert sum(x['received_cash']for x in flows if x['scenario_id']=='normal')==amount
        assert rows[v]['independent_cash_oracle'][0]['counterparty_receipts']==0
