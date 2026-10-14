/* Typed contracts mirroring the FastAPI backend responses. */

export interface BackendHealth {
  status: string;
  service: string;
  version: string;
  time: string;
  torch_version: string;
  device: string;
  cuda_available: boolean;
  dataset_file_exists: boolean;
  dataset_loaded: boolean;
  active_job_id: string | null;
}

export interface WebHealth {
  status: "ok";
  app: string;
  time: string;
  backend: { reachable: boolean; url: string; detail: BackendHealth | null; error?: string };
}

export interface DatasetFeature {
  index: number;
  key: string;
  name: string;
  unit: string;
  role: string;
}

export interface DownloadState {
  in_progress: boolean;
  bytes_downloaded: number;
  total_bytes: number | null;
  error: string | null;
  percent: number | null;
  last_result: string | null;
}

export interface SplitPartition {
  start_index: number;
  end_index: number;
  num_timesteps: number;
  start_timestamp: string | null;
  end_timestamp: string | null;
}

export interface SplitInfo {
  strategy: string;
  ratios: { train: number; val: number; test: number };
  partitions: { train: SplitPartition; val: SplitPartition; test: SplitPartition };
}

export type LoadStatus = "not_downloaded" | "loaded" | "error";

export interface DatasetStatus {
  name: string;
  source: string;
  source_repository: string;
  source_url: string;
  citation: string;
  local_path: string;
  file_exists: boolean;
  file_size_bytes: number | null;
  expected_shape: number[];
  sampling_interval_minutes: number;
  features: DatasetFeature[];
  flow_channel: number;
  download: DownloadState;
  manual_instructions: string;
  load_status: LoadStatus;
  error: string | null;
  shape: number[] | null;
  num_timesteps: number | null;
  num_sensors: number | null;
  num_features: number | null;
  archive_keys?: string[];
  array_key?: string;
  original_dtype?: string;
  in_memory_dtype?: string;
  in_memory_bytes?: number;
  shape_matches_documentation?: boolean;
  missing_values?: { nan_count: number; zero_flow_count: number; note: string };
  time_range?: { derived: boolean; start: string | null; end: string | null; basis: string };
  split?: SplitInfo;
  duration_days?: number;
}

export interface SensorFeatureStats {
  key: string;
  name: string;
  unit: string;
  mean: number | null;
  std: number | null;
  min: number | null;
  max: number | null;
  nan_count: number;
  zero_count: number;
}

export interface SensorStatistics {
  sensor_id: number;
  num_timesteps: number;
  features: SensorFeatureStats[];
  train_period_flow: { mean: number | null; std: number | null };
  coordinates_available: boolean;
  coordinates_note: string;
}

export interface SensorSeries {
  sensor_id: number;
  start: number;
  end: number;
  stride: number;
  downsampled: boolean;
  num_points: number;
  sampling_interval_minutes: number;
  indices: number[];
  timestamps: string[] | null;
  flow: number[];
  occupancy?: number[];
  speed?: number[];
  statistics: SensorStatistics;
}

export interface SensorsOverview {
  num_sensors: number;
  train_mean_flow: number[];
  train_std_flow: number[];
  basis: string;
}

export interface ClientSummary {
  client_id: number;
  name: string;
  num_examples: number;
  num_sensors: number;
  sensor_ids_preview: number[];
  sensor_ids_truncated: boolean;
  mean_flow: number;
  std_flow: number;
  strategy: string;
  windows_available?: number;
  sensor_mean_flow_min?: number;
  sensor_mean_flow_max?: number;
}

export interface PartitionPreview {
  strategy: string;
  description: string;
  num_clients: number;
  seed: number;
  clients: ClientSummary[];
  sensor_train_mean_flow: number[];
  sensor_client_assignment: number[] | null;
  basis: string;
}

export type Method = "centralized" | "fed_iid" | "fed_noniid" | "comparison";
export type RunMethod = Exclude<Method, "comparison">;
export type SeedMode = "single" | "three_seed";

export interface ExperimentConfig {
  dataset: "PEMS04";
  method: Method;
  num_clients: number;
  rounds: number;
  local_epochs: number;
  learning_rate: number;
  batch_size: number;
  seed_mode: SeedMode;
  seed: number;
  model_type: "gru";
  hidden_size: number;
  num_layers: number;
  lookback: number;
  horizon: number;
  max_train_examples: number | null;
  max_eval_examples: number | null;
  label: string | null;
}

export interface ConfigDefaults {
  quick: ExperimentConfig;
  full: ExperimentConfig;
  three_seeds: number[];
  methods: { value: Method; label: string }[];
  limits: Record<string, { min: number; max: number }>;
  split: { train: number; val: number; test: number };
  eval_sample_seed: number;
  device: string;
  workflow_steps: { key: string; label: string }[];
}

export type RunStatus = "queued" | "running" | "completed" | "failed" | "cancelled";

export interface ClientProgress {
  client_id: number;
  name: string;
  num_examples: number;
  num_sensors: number;
  mean_flow: number;
  status: string;
  epoch: number;
  total_epochs: number;
  batch: number;
  total_batches: number;
  train_loss: number | null;
  download_bytes: number;
  upload_bytes: number;
  upload_status: string;
}

export interface ServerProgress {
  status: string;
  round?: number;
  epoch?: number;
  total_epochs?: number;
  batch?: number;
  total_batches?: number;
  train_loss?: number | null;
  val_mae?: number | null;
  val_rmse?: number | null;
  aggregation?: string;
  parameter_count?: number;
  payload_bytes?: number;
  cumulative_bytes?: number;
}

export interface RoundRecord {
  round: number;
  val_mae: number;
  val_rmse: number;
  val_mse_normalized?: number;
  train_loss_mean: number;
  num_clients: number;
  total_download_bytes: number;
  total_upload_bytes: number;
  round_bytes: number;
  cumulative_bytes: number;
  duration_seconds: number;
  timestamp: string;
  download_bytes_per_client?: number[];
  upload_bytes_per_client?: number[];
  bytes_per_client?: number[];
  client_train_losses?: number[];
  client_num_examples?: number[];
  aggregation_weights?: number[];
  epochs_completed?: number;
  epoch_losses?: number[];
  model_parameter_count?: number;
}

export interface RunProgress {
  status?: RunStatus;
  job_id?: string;
  phase: string;
  message?: string;
  percent: number;
  current_round: number;
  total_rounds: number;
  workflow_step?: string;
  server: ServerProgress;
  clients: ClientProgress[];
  history: RoundRecord[];
}

export interface JobRun {
  experiment_id: string;
  method: RunMethod;
  method_label: string;
  seed: number;
  status: RunStatus;
  progress: Partial<RunProgress>;
  error: string | null;
  started_at: string | null;
  finished_at: string | null;
}

export interface Job {
  job_id: string;
  status: string;
  label: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  error: string | null;
  config: ExperimentConfig;
  seeds: number[];
  methods: RunMethod[];
  runs: JobRun[];
  current_index: number;
  runs_total: number;
  runs_finished: number;
  percent: number;
  cancel_requested: boolean;
}

export interface CommunicationSummary {
  type: "federated" | "centralized";
  measurement: string;
  note: string;
  total_bytes: number;
  rounds?: number;
  num_clients?: number;
  model_payload_bytes_per_transfer?: number;
  total_download_bytes?: number;
  total_upload_bytes?: number;
  bytes_per_round_mean?: number;
  estimated_raw_data_upload_bytes?: number;
  deduplicated_series_upload_bytes?: number;
  unique_observations_covered?: number;
  num_examples?: number;
  values_per_example?: number;
  itemsize_bytes?: number;
  measured_model_transfer_bytes?: number;
}

export interface PredictionSample {
  sensor_id: number;
  window_start: number;
  forecast_origin: number;
  forecast_origin_timestamp: string | null;
  input_flow: number[];
  actual: number[];
  predicted: number[];
  mae: number;
}

export interface CommunicationBrief {
  type: string | null;
  total_bytes: number | null;
  total_download_bytes: number | null;
  total_upload_bytes: number | null;
  estimated_raw_data_upload_bytes: number | null;
  model_payload_bytes_per_transfer: number | null;
}

export interface ExperimentSummary {
  experiment_id: string;
  job_id: string | null;
  label: string | null;
  status: RunStatus;
  method: RunMethod;
  method_label: string;
  seed: number;
  seed_mode: SeedMode;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
  duration_seconds: number | null;
  config: Partial<ExperimentConfig>;
  partition_strategy: string | null;
  train_examples: number | null;
  train_sampled: boolean | null;
  test_examples: number | null;
  rounds_completed: number;
  final_val_mae: number | null;
  final_val_rmse: number | null;
  test_mae: number | null;
  test_rmse: number | null;
  communication: CommunicationBrief;
  model_parameter_count: number | null;
  error: string | null;
}

export interface ExperimentRecord {
  experiment_id: string;
  job_id: string | null;
  label: string | null;
  status: RunStatus;
  method: RunMethod;
  method_label: string;
  seed: number;
  seed_mode: SeedMode;
  created_at: string;
  started_at?: string;
  finished_at?: string;
  duration_seconds?: number;
  config: ExperimentConfig;
  device?: string;
  dataset?: {
    name: string;
    source_url: string;
    shape: number[];
    sampling_interval_minutes: number;
    split: SplitInfo;
    normalization: { method: string; mean: number[]; std: number[] };
    preprocessing_note: string;
  };
  data?: {
    train_examples: number;
    train_examples_available: number;
    train_sampled: boolean;
    val_examples: number;
    val_examples_available: number;
    test_examples: number;
    test_examples_available: number;
    eval_sampled: boolean;
    eval_sample_seed: number;
    sampling_note: string;
  };
  model?: {
    type: string;
    input_size: number;
    hidden_size: number;
    num_layers: number;
    horizon: number;
    lookback: number;
    parameter_count: number;
    state_dict_bytes: number;
    raw_parameter_bytes: number;
    input_minutes: number;
    horizon_minutes: number;
    loss: string;
    optimizer: string;
  };
  partitioning?: {
    strategy: string;
    description: string;
    num_clients?: number;
    total_examples?: number;
    clients: ClientSummary[];
    uses_test_data?: boolean;
  };
  history: RoundRecord[];
  test_metrics?: { mae: number; rmse: number; num_examples: number };
  best_validation?: { round: number; val_mae: number; val_rmse: number } | null;
  communication?: CommunicationSummary;
  predictions_sample?: PredictionSample[];
  model_path?: string;
  round_definition?: string;
  progress?: RunProgress;
  error: string | null;
  traceback?: string;
}

export interface PredictionSeries {
  experiment_id: string;
  method: RunMethod;
  sensor_id: number;
  horizon_step: number;
  horizon_minutes: number;
  test_period: { start_index: number; end_index: number; first_origin: number; last_origin: number };
  start: number;
  steps: number;
  indices: number[];
  timestamps: string[] | null;
  actual: number[];
  predicted: number[];
  mae: number;
  rmse: number;
}

export interface CreateExperimentResponse {
  job: Job;
  experiment_ids: string[];
}
