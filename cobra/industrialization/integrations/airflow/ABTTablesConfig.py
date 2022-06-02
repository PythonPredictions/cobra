import os
from typing import Dict, List


class ABTTablesConfig:
    """
    Class to abstract the configuration and naming convention of tables and
    datasets in ABT
    """

    def __init__(self):
        self.dataset_slg_sds = os.getenv("ABT_DATASET_SLG_SDS", "slg_sds_dev")
        self.dataset_slg_sds_hist = os.getenv("ABT_DATASET_SLG_SDS_HIST", "slg_sds_hist_dev")
        self.input_dataset = os.getenv("ABT_DATASET_INPUT", "input_dev")
        self.dataset_intermediate = os.getenv("ABT_DATASET_INTERMEDIATE", "intermediate_dev")
        self.dataset_output = os.getenv("ABT_DATASET_OUTPUT", "output_dev")
        self.source_project = os.getenv("ABT_SOURCE_PROJECT", "analytical-base-tables-staging")
        self.target_project = os.getenv("ABT_TARGET_PROJECT", "analytical-base-tables-staging")
        self.bucket_backups = os.getenv("ABT_BUCKET_BACKUPS", "abt_backups_dev")
        self.location = os.getenv("ABT_LOCATION", "eu")
        self.run_date_format = "%Y%m%d"

    def get_source_tables(self) -> List[Dict[str,str]]:
        return [
            {"dataset": self.dataset_slg_sds, "table": "communication_stats"},
            {"dataset": self.dataset_slg_sds, "table": "contact_moments"},
            {"dataset": self.dataset_slg_sds, "table": "site_tag_product_hits"},
            {"dataset": self.dataset_slg_sds, "table": "users_maxeda"},
            {"dataset": self.dataset_slg_sds_hist, "table": "COMMUNICATIONDOMAIN"},
            {"dataset": self.dataset_slg_sds_hist, "table": "COMMUNICATIONS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "CONTACTS_TAX_EMAIL"},
            {"dataset": self.dataset_slg_sds_hist, "table": "CONTACTS_TAX_TRANS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "FAV_STORE"},
            {"dataset": self.dataset_slg_sds_hist, "table": "INTERACTIONS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "MAILCLIENTCODES"},
            {"dataset": self.dataset_slg_sds_hist, "table": "MESSAGE"},
            {"dataset": self.dataset_slg_sds_hist, "table": "MESSAGEDELIVERYSTATES"},
            {"dataset": self.dataset_slg_sds_hist, "table": "PROBECODES"},
            {"dataset": self.dataset_slg_sds_hist, "table": "PRODUCTS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_COMMUNICATIONDOMAIN"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_COMMUNICATIONS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_DATA_CONSENT"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_FAV_STORE"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_INTERACTIONS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_LOYALTYCARDS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_MAILCLIENTCODES"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_MESSAGE"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_MESSAGEDELIVERYSTATES"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_MOBILE_BRICO"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_MOBILE_PRAXIS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_PROBECODES"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_PRODUCTS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_RFM_BRICO"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_RFM_PRAXIS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_SHOPS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_SITETAGPRODUCT"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_SUBJECTRIGHTS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_TRANSACTIONLINES"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_TRANSACTIONS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RAW_USERS_CONTACTS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RFM_BRICO"},
            {"dataset": self.dataset_slg_sds_hist, "table": "RFM_PRAXIS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "SHOPS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "SITETAGPRODUCT"},
            {"dataset": self.dataset_slg_sds_hist, "table": "TRANSACTIONS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "TRANSACTIONLINES"},
            {"dataset": self.dataset_slg_sds_hist, "table": "USERS_CONTACTS"},
            {"dataset": self.dataset_slg_sds_hist, "table": "purchase_history"},
            {"dataset": self.dataset_slg_sds_hist, "table": "purchase_history_grouped"}

        ]

    def get_target_table_name(self, source_dataset: str, source_table) -> str:
        """
        Method to build a target table name based on a source dataset and source table to be used
        in the input dataset

        :param source_dataset: name of the original source_dataset
        :param source_table: name of the table in the original source_dataset
        :return: string with source_dataset and source_table
        """
        return f"{source_dataset}_{source_table}"

    def get_target_table_names(self) -> Dict[str, str]:
        """
        Method to retrieve a dictionary with the mappings of all table names in the input dataset

        :return: A dictionary mapping a table_{table name} to the name of the table in the
        input dataset
        """
        return  {
            f"table_{st['table']}": self.get_target_table_name(st['dataset'], st['table'])
            for st in self.get_source_tables()
        }
