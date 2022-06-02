CREATE TABLE IF NOT EXISTS {{project}}.{{dataset_output}}.garden_score
(
  user_id INT64,
  score FLOAT64,
  score_date DATE
)
PARTITION BY
  score_date
OPTIONS(
  require_partition_filter=true
)