/**
 * Model Management Page
 */

import { useState, useEffect, useCallback } from 'react';
import { api, type ModelVersionResponse } from '../services/api';

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleString();
}

function formatMetric(value?: number): string {
  if (value === undefined || value === null) return 'N/A';
  return (value * 100).toFixed(2) + '%';
}

export function ModelsPage() {
  const [models, setModels] = useState<ModelVersionResponse[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [activating, setActivating] = useState<number | null>(null);

  const fetchModels = useCallback(async () => {
    try {
      setIsLoading(true);
      const data = await api.getModels();
      setModels(data);
    } catch {
      // Ignore errors
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchModels();
  }, [fetchModels]);

  const handleActivate = async (modelId: number) => {
    if (window.confirm('Activate this model version? This will deactivate the current active model.')) {
      try {
        setActivating(modelId);
        await api.activateModel(modelId);
        await fetchModels();
      } catch (err: unknown) {
        const axiosError = err as { response?: { data?: { detail?: string } } };
        alert(axiosError.response?.data?.detail || 'Failed to activate model');
      } finally {
        setActivating(null);
      }
    }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Model Management</h1>
          <p className="text-gray-500 mt-1">View and manage model versions</p>
        </div>
        <button onClick={fetchModels} disabled={isLoading} className="btn-secondary">
          {isLoading ? 'Refreshing...' : 'Refresh'}
        </button>
      </div>

      <div className="card">
        {models.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <p>No model versions found.</p>
          </div>
        ) : (
          <div className="table-container">
            <table className="table">
              <thead>
                <tr>
                  <th>Version</th>
                  <th>Model</th>
                  <th>Type</th>
                  <th>Training Date</th>
                  <th>Dataset Size</th>
                  <th>Accuracy</th>
                  <th>Precision</th>
                  <th>Recall</th>
                  <th>F1 Score</th>
                  <th>ROC AUC</th>
                  <th>PR AUC</th>
                  <th>Status</th>
                  <th>Actions</th>
                </tr>
              </thead>
              <tbody className="bg-white divide-y divide-gray-200">
                {models.map((model) => (
                  <tr key={model.id} className={model.is_active ? 'bg-primary-50' : 'hover:bg-gray-50'}>
                    <td className="font-mono text-sm">{model.version}</td>
                    <td className="font-medium">{model.model_name}</td>
                    <td className="text-sm">{model.model_type}</td>
                    <td className="text-sm text-gray-500">{formatDate(model.training_date)}</td>
                    <td className="text-sm">{model.training_dataset_size?.toLocaleString() || 'N/A'}</td>
                    <td className="text-sm font-mono">{formatMetric(model.accuracy)}</td>
                    <td className="text-sm font-mono">{formatMetric(model.precision)}</td>
                    <td className="text-sm font-mono">{formatMetric(model.recall)}</td>
                    <td className="text-sm font-mono">{formatMetric(model.f1_score)}</td>
                    <td className="text-sm font-mono">{formatMetric(model.roc_auc)}</td>
                    <td className="text-sm font-mono">{formatMetric(model.pr_auc)}</td>
                    <td>
                      <span className={model.is_active ? 'badge-high' : 'badge-pending'}>
                        {model.is_active ? 'Active' : model.status.charAt(0).toUpperCase() + model.status.slice(1)}
                      </span>
                    </td>
                    <td>
                      <div className="flex items-center space-x-2">
                        {!model.is_active && (
                          <button
                            onClick={() => handleActivate(model.id)}
                            disabled={activating === model.id}
                            className="text-primary-600 hover:text-primary-700 text-sm font-medium"
                          >
                            {activating === model.id ? 'Activating...' : 'Activate'}
                          </button>
                        )}
                        {model.is_active && (
                          <span className="text-green-600 text-sm font-medium">● Active</span>
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

      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Current Active Model</h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {models.filter(m => m.is_active).map((model) => (
            <div key={model.id} className="bg-primary-50 border border-primary-200 rounded-lg p-4">
              <h3 className="font-semibold text-gray-900">{model.model_name} v{model.version}</h3>
              <p className="text-sm text-gray-500 mt-1">{model.model_type}</p>
              <div className="mt-4 grid grid-cols-2 gap-4 text-sm">
                <div>
                  <p className="text-gray-500">Accuracy</p>
                  <p className="font-mono">{formatMetric(model.accuracy)}</p>
                </div>
                <div>
                  <p className="text-gray-500">F1 Score</p>
                  <p className="font-mono">{formatMetric(model.f1_score)}</p>
                </div>
                <div>
                  <p className="text-gray-500">ROC AUC</p>
                  <p className="font-mono">{formatMetric(model.roc_auc)}</p>
                </div>
                <div>
                  <p className="text-gray-500">PR AUC</p>
                  <p className="font-mono">{formatMetric(model.pr_auc)}</p>
                </div>
              </div>
              <p className="mt-4 text-xs text-gray-500">
                Trained: {formatDate(model.training_date)}
              </p>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}