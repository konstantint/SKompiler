import sqlalchemy as sa
from skompiler.dsl import *

def equal_queries(query1, query2):
    assert ''.join(query1.strip().lower().split()) == ''.join(query2.strip().lower().split())

def test_from_obj():
    expr = ident('a')*ident('b')

    result = expr.to('sqlalchemy/sqlite', key_column='_key_', from_obj='_table_')
    equal_queries(result, 'select a * b as y from _table_')

    result = expr.to('sqlalchemy/sqlite', key_column='_key_', from_obj=sa.table('_table_', sa.column('_key_')))
    equal_queries(result, 'select a * b as y from _table_')

    cte = sa.select([sa.column('_key_'), (sa.column('x')*2).label('a')], from_obj=sa.text('_table_')).cte('_cte_')
    result = expr.to('sqlalchemy/sqlite', key_column='_key_', from_obj=cte)
    equal_queries(result, 'with _cte_ as (select _key_, x*2 as a from _table_) select a * b as y from _cte_')
