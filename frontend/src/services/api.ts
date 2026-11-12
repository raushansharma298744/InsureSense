/**
 * API Service for InsureSense Frontend
 */

import axios, { AxiosInstance, InternalAxiosRequestConfig, AxiosError } from 'axios';
import type {
  LoginRequest,
  RegisterRequest,
  TokenResponse,
  RefreshTokenRequest,
  User,
  PredictionRequest,
  PredictionResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
  PredictionHistoryResponse,
  PredictionHistoryItem,
  PredictionFilter,
  BatchJobCreateResponse,
  BatchJobResponse,
  BatchJobListResponse,
  ModelVersionResponse,
  DashboardSummaryResponse,
  BatchJobStatus,
  PropensityCategory,
} from '../types';

const getApiBaseUrl = (): string => {
  const envUrl = (import.meta as unknown as { env: { VITE_API_URL?: string } }).env?.VITE_API_URL;
  const baseUrl = (envUrl || 'http://localhost:8000').trim().replace(/\/$/, '');
  return baseUrl.endsWith('/api/v1') ? baseUrl : `${baseUrl}/api/v1`;
};

const API_BASE_URL = getApiBaseUrl();

class ApiService {
  private client: AxiosInstance;
  private accessToken: string | null = null;
  private refreshToken: string | null = null;

  constructor() {
    this.client = axios.create({
      baseURL: API_BASE_URL,
      headers: {
        'Content-Type': 'application/json',
      },
      timeout: 30000,
    });

    this.client.interceptors.request.use(
      (config: InternalAxiosRequestConfig) => {
        if (this.accessToken) {
          config.headers.Authorization = `Bearer ${this.accessToken}`;
        }
        return config;
      },
      (error) => Promise.reject(error)
    );

    this.client.interceptors.response.use(
      (response) => response,
      async (error: AxiosError) => {
        const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

        if (error.response?.status === 401 && !originalRequest._retry && this.refreshToken) {
          originalRequest._retry = true;
          try {
            await this.refreshAccessToken();
            if (this.accessToken && originalRequest.headers) {
              originalRequest.headers.Authorization = `Bearer ${this.accessToken}`;
            }
            return this.client(originalRequest);
          } catch {
            this.clearTokens();
            window.location.href = '/login';
            return Promise.reject(error);
          }
        }

        return Promise.reject(error);
      }
    );

    this.loadTokens();
  }

  private loadTokens(): void {
    this.accessToken = localStorage.getItem('access_token');
    this.refreshToken = localStorage.getItem('refresh_token');
  }

  private saveTokens(access: string, refresh: string): void {
    this.accessToken = access;
    this.refreshToken = refresh;
    localStorage.setItem('access_token', access);
    localStorage.setItem('refresh_token', refresh);
  }

  private clearTokens(): void {
    this.accessToken = null;
    this.refreshToken = null;
    localStorage.removeItem('access_token');
    localStorage.removeItem('refresh_token');
  }

  private async refreshAccessToken(): Promise<void> {
    if (!this.refreshToken) throw new Error('No refresh token');

    const response = await axios.post<TokenResponse>(
      `${API_BASE_URL}/auth/refresh`,
      { refresh_token: this.refreshToken },
      { headers: { 'Content-Type': 'application/json' } }
    );

    this.saveTokens(response.data.access_token, response.data.refresh_token);
  }

  // Auth methods
  async login(credentials: LoginRequest): Promise<TokenResponse> {
    const response = await this.client.post<TokenResponse>('/auth/login', credentials);
    this.saveTokens(response.data.access_token, response.data.refresh_token);
    return response.data;
  }

  async register(userData: RegisterRequest): Promise<User> {
    const response = await this.client.post<User>('/auth/register', userData);
    return response.data;
  }

  async getCurrentUser(): Promise<User> {
    const response = await this.client.get<User>('/auth/me');
    return response.data;
  }

  async updateProfile(data: Partial<User>): Promise<User> {
    const response = await this.client.put<User>('/auth/me', data);
    return response.data;
  }

  logout(): void {
    this.clearTokens();
  }

  isAuthenticated(): boolean {
    return !!this.accessToken;
  }

  // Prediction methods
  async createPrediction(request: PredictionRequest, threshold = 0.5): Promise<PredictionResponse> {
    const response = await this.client.post<PredictionResponse>('/predictions', request, {
      params: { threshold },
    });
    return response.data;
  }

  async createBatchPredictions(request: BatchPredictionRequest, threshold = 0.5): Promise<BatchPredictionResponse> {
    const response = await this.client.post<BatchPredictionResponse>('/predictions/batch', request, {
      params: { threshold },
    });
    return response.data;
  }

  async uploadBatchFile(file: File, threshold = 0.5): Promise<BatchJobCreateResponse> {
    const formData = new FormData();
    formData.append('file', file);
    formData.append('threshold', threshold.toString());

    const response = await this.client.post<BatchJobCreateResponse>('/predictions/batch/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
  }

  async getPredictionHistory(
    filters: PredictionFilter = {},
    page = 1,
    pageSize = 50
  ): Promise<PredictionHistoryResponse> {
    const params = new URLSearchParams();
    Object.entries(filters).forEach(([key, value]) => {
      if (value !== undefined && value !== null) {
        params.append(key, String(value));
      }
    });
    params.append('skip', String((page - 1) * pageSize));
    params.append('limit', String(pageSize));

    const response = await this.client.get<PredictionHistoryResponse>(`/predictions/history?${params.toString()}`);
    return response.data;
  }

  async getPredictionDetail(id: number): Promise<PredictionHistoryResponse['items'][0]> {
    const response = await this.client.get(`/predictions/history/${id}`);
    return response.data;
  }

  // Batch Job methods
  async getBatchJobs(
    status?: string,
    page = 1,
    pageSize = 50
  ): Promise<BatchJobListResponse> {
    const params = new URLSearchParams();
    if (status) params.append('status', status);
    params.append('skip', String((page - 1) * pageSize));
    params.append('limit', String(pageSize));

    const response = await this.client.get<BatchJobListResponse>(`/predictions/batch/jobs?${params.toString()}`);
    return response.data;
  }

  async getBatchJob(jobId: number): Promise<BatchJobResponse> {
    const response = await this.client.get<BatchJobResponse>(`/predictions/batch/jobs/${jobId}`);
    return response.data;
  }

  async downloadBatchResults(jobId: number): Promise<Blob> {
    const response = await this.client.get(`/predictions/batch/jobs/${jobId}/download`, {
      responseType: 'blob',
    });
    return response.data;
  }

  // Model methods
  async getModels(page = 1, pageSize = 50): Promise<ModelVersionResponse[]> {
    const response = await this.client.get<ModelVersionResponse[]>(`/models?skip=${(page - 1) * pageSize}&limit=${pageSize}`);
    return response.data;
  }

  async getActiveModel(): Promise<ModelVersionResponse> {
    const response = await this.client.get<ModelVersionResponse>('/models/active');
    return response.data;
  }

  async activateModel(modelId: number): Promise<ModelVersionResponse> {
    const response = await this.client.post<ModelVersionResponse>(`/models/${modelId}/activate`);
    return response.data;
  }

  // Dashboard methods
  async getDashboardSummary(): Promise<DashboardSummaryResponse> {
    const response = await this.client.get<DashboardSummaryResponse>('/dashboard/summary');
    return response.data;
  }

  // Health check
  async healthCheck(): Promise<{ status: string; service: string; version: string }> {
    const response = await this.client.get('/health');
    return response.data;
  }
}

export const api = new ApiService();
export default api;

// Re-export types for convenience
export type {
  LoginRequest,
  RegisterRequest,
  TokenResponse,
  RefreshTokenRequest,
  User,
  PredictionRequest,
  PredictionResponse,
  BatchPredictionRequest,
  BatchPredictionResponse,
  PredictionHistoryResponse,
  PredictionHistoryItem,
  PredictionFilter,
  BatchJobCreateResponse,
  BatchJobResponse,
  BatchJobListResponse,
  ModelVersionResponse,
  DashboardSummaryResponse,
  BatchJobStatus,
  PropensityCategory,
};