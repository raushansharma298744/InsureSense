/**
 * Type definitions for InsureSense Frontend
 */

// Auth types
export interface User {
  id: number;
  email: string;
  full_name: string | null;
  role: 'admin' | 'employee';
  is_active: boolean;
  is_superuser: boolean;
  created_at: string;
  last_login: string | null;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  password: string;
  full_name?: string;
  role?: 'admin' | 'employee';
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface RefreshTokenRequest {
  refresh_token: string;
}

// Prediction types
export type PropensityCategory = 'high' | 'medium' | 'low';
export type VehicleAge = '< 1 Year' | '1-2 Year' | '> 2 Years';
export type VehicleDamage = 'No' | 'Yes';
export type Gender = 'Male' | 'Female';

export interface PredictionRequest {
  gender: Gender;
  age: number;
  driving_license: 0 | 1;
  region_code: number;
  previously_insured: 0 | 1;
  vehicle_age: VehicleAge;
  vehicle_damage: VehicleDamage;
  annual_premium: number;
  policy_sales_channel: number;
  vintage: number;
}

export interface PredictionResponse {
  prediction: 0 | 1;
  probability: number;
  propensity_category: PropensityCategory;
  threshold_used: number;
  model_version?: string;
}

export interface BatchPredictionRequest {
  customers: PredictionRequest[];
}

export interface BatchPredictionResponse {
  predictions: PredictionResponse[];
  total: number;
  successful: number;
  failed: number;
}

export interface PredictionHistoryItem {
  id: number;
  gender: string;
  age: number;
  vehicle_age: string;
  vehicle_damage: string;
  annual_premium: number;
  predicted_response: 0 | 1;
  purchase_probability: number;
  propensity_category: PropensityCategory;
  model_version?: string;
  created_at: string;
}

export interface PredictionHistoryResponse {
  items: PredictionHistoryItem[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

export interface PredictionFilter {
  propensity_category?: PropensityCategory;
  predicted_response?: 0 | 1;
  start_date?: string;
  end_date?: string;
  min_probability?: number;
  max_probability?: number;
}

// Batch Job types
export type BatchJobStatus = 'pending' | 'processing' | 'completed' | 'failed';

export interface BatchJobCreateResponse {
  job_id: number;
  status: BatchJobStatus;
  message: string;
}

export interface BatchJobResponse {
  id: number;
  filename: string;
  total_rows: number;
  processed_rows: number;
  failed_rows: number;
  status: BatchJobStatus;
  error_message?: string;
  result_file_path?: string;
  model_version?: string;
  created_at: string;
  started_at?: string;
  completed_at?: string;
}

export interface BatchJobListResponse {
  items: BatchJobResponse[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// Model types
export interface ModelVersionResponse {
  id: number;
  version: string;
  model_name: string;
  model_type: string;
  training_date: string;
  training_dataset_size?: number;
  accuracy?: number;
  precision?: number;
  recall?: number;
  f1_score?: number;
  roc_auc?: number;
  pr_auc?: number;
  is_active: boolean;
  status: string;
  created_at: string;
  activated_at?: string;
  notes?: string;
}

// Dashboard types
export interface DashboardSummaryResponse {
  total_customers_analyzed: number;
  total_predictions: number;
  total_batch_jobs: number;
  predicted_purchase_distribution: Record<string, number>;
  propensity_distribution: Record<string, number>;
  recent_predictions: number;
  recent_batch_jobs: number;
  model_info?: ModelVersionResponse;
}

// API Error type
export interface ApiError {
  detail: string;
}