import pytest

import pandas as pd
from cobra.evaluation.pigs_tables import generate_pig_tables

class TestPigTablesGeneration:

    @pytest.mark.parametrize("id_col_name", [None, "col_id"]) # test None as this is the default value in generate pig tabels 
    def test_col_id(self, id_col_name):
        
        # input
        data = pd.DataFrame({
            'col_id': [0, 1, 3, 4, 6],
            'survived': [0, 1, 1, 0, 0],
            'pclass': [3, 1, 1, 3, 1],
            'sex': ['male', 'female', 'female', 'male', 'male'],
            'age': [22.0, 38.0, 35.0, 35.0, 54.0]
        })
        target = "survived"
        prep_col = ["pclass", "sex", "age"]
        
        # output
        out = generate_pig_tables(
            basetable= data,
            target_column_name=target,
            preprocessed_predictors=prep_col,
            id_column_name=id_col_name
        )
        
        # expected
        expected = pd.DataFrame({
            'variable': ['age', 'age', 'age', 'age', 'pclass', 'pclass', 'sex', 'sex'],
            'label': [22.0, 35.0, 38.0, 54.0, 1, 3, 'female', 'male'],
            'pop_size': [0.2, 0.4, 0.2, 0.2, 0.6, 0.4, 0.4, 0.6],
            'global_avg_target': [0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4, 0.4],
            'avg_target': [0.0, 0.5, 1.0, 0.0, 0.6666666666666666, 0.0, 1.0, 0.0]
        })

        pd.testing.assert_frame_equal(out, expected)