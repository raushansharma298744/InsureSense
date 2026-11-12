/**
 * Prediction History Page
 */

import { useState, useEffect, useCallback } from 'react';
import { api, type PredictionFilter, type PredictionHistoryItem } from '../services/api';

function formatDate(dateString: string): string {
  return new Date(dateString).toLocaleString();
}

function formatProbability(prob: number): string {
  return `${(prob * 100).toFixed(1)}%`;
}

const PROPENSITY_STYLES: Record<string, string> = {
  high: 'badge-high',
  medium: 'badge-medium',
  low: 'badge-low',
};

export function HistoryPage() {
  const [history, setHistory] = useState<PredictionHistoryItem[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [pageSize] = useState(20);
  const [totalPages, setTotalPages] = useState(0);
  const [filters, setFilters] = useState<PredictionFilter>({});
  const [selectedItem, setSelectedItem] = useState<PredictionHistoryItem | null>(null);

  const fetchHistory = useCallback(async () => {
    try {
      const response = await api.getPredictionHistory(filters, page, pageSize);
      setHistory(response.items);
      setTotal(response.total);
      setTotalPages(response.total_pages);
    } catch {
      // Ignore fetch errors
    }
  }, [filters, page, pageSize]);

  useEffect(() => {
    fetchHistory();
  }, [fetchHistory]);

  const handleFilterChange = (newFilters: Partial<PredictionFilter>) => {
    setFilters(prev => ({ ...prev, ...newFilters }));
    setPage(1);
  };

  const handleClearFilters = () => {
    setFilters({});
    setPage(1);
  };

  const handlePageChange = (newPage: number) => {
    if (newPage >= 1 && newPage <= totalPages) {
      setPage(newPage);
    }
  };

  const handleViewDetail = (item: PredictionHistoryItem) => {
    setSelectedItem(item);
  };

  const handleCloseModal = () => {
    setSelectedItem(null);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Prediction History</h1>
          <p className="text-gray-500 mt-1">View and filter past predictions</p>
        </div>
        <div className="flex items-center space-x-4">
          {Object.keys(filters).length > 0 && (
            <button onClick={handleClearFilters} className="btn-secondary text-sm">
              Clear Filters
            </button>
          )}
        </div>
      </div>

      <div className="card">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
          <div>
            <label className="label">Propensity Category</label>
            <select
              value={filters.propensity_category || ''}
              onChange={(e) => handleFilterChange({ propensity_category: e.target.value as 'high' | 'medium' | 'low' | undefined })}
              className="input"
            >
              <option value="">All Categories</option>
              <option value="high">High</option>
              <option value="medium">Medium</option>
              <option value="low">Low</option>
            </select>
          </div>

          <div>
            <label className="label">Prediction</label>
            <select
              value={filters.predicted_response?.toString() || ''}
              onChange={(e) => handleFilterChange({ predicted_response: e.target.value ? parseInt(e.target.value, 10) as 0 | 1 : undefined })}
              className="input"
            >
              <option value="">All Predictions</option>
              <option value="1">Will Purchase (1)</option>
              <option value="0">No Purchase (0)</option>
            </select>
          </div>

          <div>
            <label className="label">Start Date</label>
            <input
              type="date"
              value={filters.start_date || ''}
              onChange={(e) => handleFilterChange({ start_date: e.target.value || undefined })}
              className="input"
            />
          </div>

          <div>
            <label className="label">End Date</label>
            <input
              type="date"
              value={filters.end_date || ''}
              onChange={(e) => handleFilterChange({ end_date: e.target.value || undefined })}
              className="input"
            />
          </div>

          <div>
            <label className="label">Min Probability</label>
            <input
              type="number"
              step="0.01"
              min="0"
              max="1"
              value={filters.min_probability?.toString() || ''}
              onChange={(e) => handleFilterChange({ min_probability: e.target.value ? parseFloat(e.target.value) : undefined })}
              className="input"
              placeholder="0.00"
            />
          </div>

          <div className="lg:col-span-5 flex items-end justify-end">
            <span className="text-sm text-gray-500">
              Showing {(page - 1) * pageSize + 1} to {Math.min(page * pageSize, total)} of {total} results
            </span>
          </div>
        </div>
      </div>

      <div className="card">
        {history.length === 0 ? (
          <div className="text-center py-12 text-gray-500">
            <p>No predictions found matching the criteria.</p>
          </div>
        ) : (
          <>
            <div className="table-container">
              <table className="table">
                <thead>
                  <tr>
                    <th>ID</th>
                    <th>Gender</th>
                    <th>Age</th>
                    <th>Vehicle Age</th>
                    <th>Vehicle Damage</th>
                    <th>Annual Premium</th>
                    <th>Prediction</th>
                    <th>Probability</th>
                    <th>Propensity</th>
                    <th>Date</th>
                    <th>Actions</th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {history.map((item) => (
                    <tr key={item.id} className="hover:bg-gray-50 cursor-pointer" onClick={() => handleViewDetail(item)}>
                      <td className="font-mono text-sm">#{item.id}</td>
                      <td className="text-sm">{item.gender}</td>
                      <td className="text-sm">{item.age}</td>
                      <td className="text-sm">{item.vehicle_age}</td>
                      <td className="text-sm">{item.vehicle_damage}</td>
                      <td className="text-sm">${item.annual_premium.toLocaleString()}</td>
                      <td className="text-sm font-medium">
                        {item.predicted_response === 1 ? 'Purchase' : 'No Purchase'}
                      </td>
                      <td className="text-sm font-mono">{formatProbability(item.purchase_probability)}</td>
                      <td>
                        <span className={PROPENSITY_STYLES[item.propensity_category]}>
                          {item.propensity_category.charAt(0).toUpperCase() + item.propensity_category.slice(1)}
                        </span>
                      </td>
                      <td className="text-sm text-gray-500">{formatDate(item.created_at)}</td>
                      <td className="text-sm text-primary-600 hover:text-primary-700 font-medium">View</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            {totalPages > 1 && (
              <div className="mt-4 flex items-center justify-between">
                <p className="text-sm text-gray-500">
                  Page {page} of {totalPages}
                </p>
                <div className="flex items-center space-x-2">
                  <button
                    onClick={() => handlePageChange(page - 1)}
                    disabled={page === 1}
                    className="btn-secondary text-sm"
                  >
                    Previous
                  </button>
                  <button
                    onClick={() => handlePageChange(page + 1)}
                    disabled={page === totalPages}
                    className="btn-secondary text-sm"
                  >
                    Next
                  </button>
                </div>
              </div>
            )}
          </>
        )}
      </div>

      {selectedItem && (
        <div className="fixed inset-0 z-50 overflow-y-auto" onClick={handleCloseModal}>
          <div className="flex min-h-full items-center justify-center p-4">
            <div className="fixed inset-0 bg-black bg-opacity-25" onClick={handleCloseModal} />
            <div className="relative w-full max-w-3xl bg-white rounded-xl shadow-xl" onClick={(e) => e.stopPropagation()}>
              <div className="flex items-center justify-between p-4 border-b border-gray-200">
                <h3 className="text-lg font-semibold text-gray-900">Prediction Details</h3>
                <button onClick={handleCloseModal} className="text-gray-400 hover:text-gray-500">
                  <svg className="h-6 w-6" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <div className="p-4 space-y-4 max-h-96 overflow-y-auto">
                <dl className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  <div className="md:col-span-2">
                    <dt className="text-sm text-gray-500">Prediction ID</dt>
                    <dd className="mt-1 font-mono">#{selectedItem.id}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Gender</dt>
                    <dd className="mt-1">{selectedItem.gender}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Age</dt>
                    <dd className="mt-1">{selectedItem.age}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Vehicle Age</dt>
                    <dd className="mt-1">{selectedItem.vehicle_age}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Vehicle Damage</dt>
                    <dd className="mt-1">{selectedItem.vehicle_damage}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Annual Premium</dt>
                    <dd className="mt-1">${selectedItem.annual_premium.toLocaleString()}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Prediction</dt>
                    <dd className="mt-1 font-medium">
                      {selectedItem.predicted_response === 1 ? 'Purchase' : 'No Purchase'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Probability</dt>
                    <dd className="mt-1 font-mono">{formatProbability(selectedItem.purchase_probability)}</dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Propensity</dt>
                    <dd className="mt-1">
                      <span className={PROPENSITY_STYLES[selectedItem.propensity_category]}>
                        {selectedItem.propensity_category.charAt(0).toUpperCase() + selectedItem.propensity_category.slice(1)}
                      </span>
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm text-gray-500">Created</dt>
                    <dd className="mt-1">{formatDate(selectedItem.created_at)}</dd>
                  </div>
                  {selectedItem.model_version && (
                    <div>
                      <dt className="text-sm text-gray-500">Model Version</dt>
                      <dd className="mt-1 text-sm">{selectedItem.model_version}</dd>
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