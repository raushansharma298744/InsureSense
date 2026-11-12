/**
 * Batch Prediction Page
 */

import { useState, useEffect, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { api, type BatchJobResponse, type BatchJobStatus } from '../services/api';

const batchSchema = z.object({
  threshold: z.number().min(0).max(1).default(0.5),
});

type BatchFormData = z.infer<typeof batchSchema>;

const STATUS_STYLES: Record<BatchJobStatus, string> = {
  pending: 'badge-pending',
  processing: 'badge-processing',
  completed: 'badge-completed',
  failed: 'badge-failed',
};

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleString();
}

function formatNumber(num: number): string {
  return num.toLocaleString();
}

export function BatchPredictionPage() {
  const [jobs, setJobs] = useState<BatchJobResponse[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [selectedJob, setSelectedJob] = useState<BatchJobResponse | null>(null);

  const {
    register,
    watch,
  } = useForm<BatchFormData>({
    resolver: zodResolver(batchSchema),
    defaultValues: { threshold: 0.5 },
  });

  const threshold = watch('threshold') ?? 0.5;

  const fetchJobs = useCallback(async () => {
    try {
      const response = await api.getBatchJobs(undefined, 1, 50);
      setJobs(response.items);
    } catch {
      // Ignore fetch errors for now
    }
  }, []);

  // Poll for job updates
  useEffect(() => {
    const interval = setInterval(() => {
      const hasProcessing = jobs.some(j => j.status === 'processing' || j.status === 'pending');
      if (hasProcessing) {
        fetchJobs();
      }
    }, 5000);

    return () => clearInterval(interval);
  }, [jobs, fetchJobs]);

  // Initial load
  useEffect(() => {
    fetchJobs();
  }, [fetchJobs]);

  const onDrop = useCallback(async (acceptedFiles: File[]) => {
    const file = acceptedFiles[0];
    if (!file.name.endsWith('.csv')) {
      setUploadError('Only CSV files are allowed');
      return;
    }

    if (file.size > 10 * 1024 * 1024) {
      setUploadError('File size exceeds 10MB limit');
      return;
    }

    setUploadError(null);
    setIsUploading(true);

    try {
      const response = await api.uploadBatchFile(file, threshold);
      setJobs(prev => [response as unknown as BatchJobResponse, ...prev]);
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } };
      setUploadError(axiosError.response?.data?.detail || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  }, [threshold]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'text/csv': ['.csv'] },
    maxSize: 10 * 1024 * 1024,
    disabled: isUploading,
  });

  const handleDownload = async (jobId: number) => {
    try {
      const blob = await api.downloadBatchResults(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `batch_predictions_${jobId}.csv`;
      a.click();
      window.URL.revokeObjectURL(url);
    } catch {
      alert('Failed to download results');
    }
  };

  const handleViewJob = (job: BatchJobResponse) => {
    setSelectedJob(job);
  };

  const handleCloseModal = () => {
    setSelectedJob(null);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Batch Prediction</h1>
          <p className="text-gray-500 mt-1">Upload CSV file for bulk customer predictions</p>
        </div>
      </div>

      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Upload CSV File</h2>

        <div className="space-y-4">
          <div>
            <label className="label">Prediction Threshold</label>
            <div className="flex items-center space-x-4">
              <input
                type="range"
                min="0"
                max="1"
                step="0.05"
                {...register('threshold', { valueAsNumber: true })}
                className="flex-1 h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer accent-primary-600"
              />
              <span className="text-sm font-medium text-gray-700 w-16 text-right">
                {(threshold * 100).toFixed(0)}%
              </span>
            </div>
          </div>

          <div
            {...getRootProps()}
            className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors ${
              isDragActive
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-primary-400'
            }`}
          >
            <input {...getInputProps()} />
            <div className="space-y-2">
              <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-4l-4 4m0 0l-4-4m4 4V4" />
              </svg>
              <p className="text-lg font-medium text-gray-900">
                {isDragActive ? 'Drop the CSV file here...' : 'Drag & drop a CSV file here, or click to select'}
              </p>
              <p className="text-sm text-gray-500">
                CSV must contain columns: gender, age, driving_license, region_code, previously_insured, vehicle_age, vehicle_damage, annual_premium, policy_sales_channel, vintage
              </p>
              <p className="text-xs text-gray-400">Max file size: 10MB</p>
            </div>
          </div>

          {uploadError && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm" role="alert">
              {uploadError}
            </div>
          )}
        </div>
      </div>

      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-900">Batch Jobs</h2>
          <button onClick={fetchJobs} className="btn-secondary text-sm">
            Refresh
          </button>
        </div>

        {jobs.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <p>No batch jobs yet. Upload a CSV file to get started.</p>
          </div>
        ) : (
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Job ID</th>
                  <th>Filename</th>
                  <th>Status</th>
                  <th>Rows</th>
                  <th>Progress</th>
                  <th>Created</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {jobs.map((job) => (
                  <tr key={job.id} className="hover:bg-gray-50">
                    <td className="font-mono text-sm">#{job.id}</td>
                    <td className="text-sm truncate max-w-xs" title={job.filename}>{job.filename}</td>
                    <td>
                      <span className={STATUS_STYLES[job.status]}>
                        {job.status.charAt(0).toUpperCase() + job.status.slice(1)}
                      </span>
                    </td>
                    <td className="text-sm">{formatNumber(job.total_rows)}</td>
                    <td>
                      <div className="w-32 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full rounded-full transition-all duration-300 ${
                            job.status === 'completed' ? 'bg-green-500' :
                            job.status === 'failed' ? 'bg-red-500' :
                            job.status === 'processing' ? 'bg-blue-500 animate-pulse' :
                            'bg-gray-400'
                          }`}
                          style={{ width: job.total_rows > 0 ? `${(job.processed_rows / job.total_rows) * 100}%` : '0%' }}
                        />
                      </div>
                      <p className="text-xs text-gray-500 mt-1">
                        {formatNumber(job.processed_rows)} / {formatNumber(job.total_rows)}
                      </p>
                    </td>
                    <td className="text-sm text-gray-500">{formatDate(job.created_at)}</td>
                    <td>
                      <div className="flex items-center space-x-2">
                        <button
                          onClick={() => handleViewJob(job)}
                          className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                        >
                          View
                        </button>
                        {job.status === 'completed' && job.result_file_path && (
                          <button
                            onClick={() => handleDownload(job.id)}
                            className="text-green-600 hover:text-green-700 text-sm font-medium"
                          >
                            Download
                          </button>
                        )}
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {selectedJob && (
        <div className="fixed inset-0 z-50 overflow-y-auto" onClick={handleCloseModal}>
          <div className="flex min-h-full items-center justify-center p-4">
            <div className="fixed inset-0 bg-black bg-opacity-25" onClick={handleCloseModal} />
            <div className="relative w-full max-w-2xl bg-white rounded-xl shadow-xl" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between p-4 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">Job Details</h3>
                <button onClick={handleCloseModal} className="text-gray-400 hover:text-gray-500">
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
                <dl className="grid grid-cols-1 gap-4">
                  <div>
                    <dt className="text-sm text-gray-500">Job ID</dt>
                    <dd className="mt-1 font-mono text-sm">#{selectedJob.id}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Filename</dt>
                    <dd className="mt-1 text-sm truncate">{selectedJob.filename}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Status</dt>
                    <dd className="mt-1">
                      <span className={STATUS_STYLES[selectedJob.status]}>
                        {selectedJob.status.charAt(0).toUpperCase() + selectedJob.status.slice(1)}
                      </span>
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Total Rows</dt>
                    <dd className="mt-1 text-sm">{formatNumber(selectedJob.total_rows)}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Processed</dt>
                    <dd className="mt-1 text-sm">{formatNumber(selectedJob.processed_rows)}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Failed</dt>
                    <dd className="mt-1 text-sm">{formatNumber(selectedJob.failed_rows)}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Created</dt>
                    <dd className="mt-1 text-sm">{formatDate(selectedJob.created_at)}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Started</dt>
                    <dd className="mt-1 text-sm">{selectedJob.started_at ? formatDate(selectedJob.started_at) : 'N/A'}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Completed</dt>
                    <dd className="mt-1 text-sm">{selectedJob.completed_at ? formatDate(selectedJob.completed_at) : 'N/A'}</dd>
                  </div>
                  {selectedJob.error_message && (
                    <div className="col-span-1">
                      <dt className="text-sm text-gray-500">Error</dt>
                      <dd className="mt-1 text-sm text-red-600">{selectedJob.error_message}</dd>
                    </div>
                  )}
                  {selectedJob.model_version && (
                    <div>
                      <dt className="text-sm text-gray-500">Model Version</dt>
                      <dd className="mt-1 text-sm">{selectedJob.model_version}</dd>
                    </div>
                  )}
                </dl>
              </div>
              <div className="flex justify-end p-4 border-t border-gray-200">
                <button onClick={handleCloseModal} className="btn-secondary">
                  Close
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}