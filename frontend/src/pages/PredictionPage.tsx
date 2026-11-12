/**
 * Single Customer Prediction Page
 */

import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { api, type PredictionRequest, type PredictionResponse } from '../services/api';

const predictionSchema = z.object({
  gender: z.enum(['Male', 'Female']),
  age: z.number().min(18, 'Age must be at least 18').max(100, 'Age must be at most 100'),
  driving_license: z.enum(['0', '1']),
  region_code: z.number().min(0, 'Region code must be positive').max(100),
  previously_insured: z.enum(['0', '1']),
  vehicle_age: z.enum(['< 1 Year', '1-2 Year', '> 2 Years']),
  vehicle_damage: z.enum(['No', 'Yes']),
  annual_premium: z.number().min(0, 'Premium must be positive').max(1000000),
  policy_sales_channel: z.number().min(0).max(200),
  vintage: z.number().min(0).max(1000),
  threshold: z.number().min(0).max(1).default(0.5),
});

type PredictionFormData = z.infer<typeof predictionSchema>;

const PROPENSITY_STYLES: Record<string, string> = {
  high: 'badge-high',
  medium: 'badge-medium',
  low: 'badge-low',
};

function PropensityBadge({ category }: { category: string }) {
  return <span className={PROPENSITY_STYLES[category]}>{category.charAt(0).toUpperCase() + category.slice(1)} Propensity</span>;
}

function ProbabilityBar({ probability }: { probability: number }) {
  const percentage = Math.round(probability * 100);
  const color = percentage >= 70 ? 'bg-green-500' : percentage >= 30 ? 'bg-yellow-500' : 'bg-red-500';

  return (
    <div className="w-full bg-gray-200 rounded-full h-4 overflow-hidden">
      <div
        className={`${color} h-full rounded-full transition-all duration-500`}
        style={{ width: `${percentage}%` }}
      />
    </div>
  );
}

export function PredictionPage() {
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const {
    register,
    handleSubmit,
    watch,
    reset,
    formState: { errors },
  } = useForm<PredictionFormData>({
    resolver: zodResolver(predictionSchema),
    defaultValues: {
      gender: 'Male',
      age: 35,
      driving_license: '1',
      region_code: 28,
      previously_insured: '0',
      vehicle_age: '1-2 Year',
      vehicle_damage: 'Yes',
      annual_premium: 35000,
      policy_sales_channel: 152,
      vintage: 150,
      threshold: 0.5,
    },
  });

  const threshold = watch('threshold') ?? 0.5;

  const onSubmit = async (data: PredictionFormData) => {
    setError(null);
    setIsLoading(true);

    try {
      const requestData: PredictionRequest = {
        ...data,
        driving_license: parseInt(data.driving_license, 10) as 0 | 1,
        previously_insured: parseInt(data.previously_insured, 10) as 0 | 1,
      };

      const response = await api.createPrediction(requestData, threshold);
      setResult(response);
    } catch (err: unknown) {
      const axiosError = err as { response?: { data?: { detail?: string } } };
      setError(axiosError.response?.data?.detail || 'Prediction failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleClear = () => {
    reset({
      gender: 'Male',
      age: 35,
      driving_license: '1',
      region_code: 28,
      previously_insured: '0',
      vehicle_age: '1-2 Year',
      vehicle_damage: 'Yes',
      annual_premium: 35000,
      policy_sales_channel: 152,
      vintage: 150,
      threshold: 0.5,
    });
    setResult(null);
    setError(null);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-gray-900">Customer Prediction</h1>
          <p className="text-gray-500 mt-1">Predict likelihood of vehicle insurance purchase</p>
        </div>
      </div>

      <div className="card">
        <h2 className="text-lg font-semibold text-gray-900 mb-4">Customer Information</h2>

        <form onSubmit={handleSubmit(onSubmit)} className="space-y-6">
          {error && (
            <div className="bg-red-50 border border-red-200 text-red-700 px-4 py-3 rounded-lg text-sm" role="alert">
              {error}
            </div>
          )}

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="gender" className="label">Gender</label>
              <select id="gender" {...register('gender')} className="input" aria-invalid={!!errors.gender}>
                <option value="Male">Male</option>
                <option value="Female">Female</option>
              </select>
              {errors.gender && <p className="mt-1 text-sm text-red-600">{errors.gender.message}</p>}
            </div>

            <div>
              <label htmlFor="age" className="label">Age</label>
              <input
                id="age"
                type="number"
                {...register('age', { valueAsNumber: true })}
                className="input"
                placeholder="35"
                min={18}
                max={100}
                aria-invalid={!!errors.age}
              />
              {errors.age && <p className="mt-1 text-sm text-red-600">{errors.age.message}</p>}
            </div>

            <div>
              <label htmlFor="driving_license" className="label">Driving License</label>
              <select id="driving_license" {...register('driving_license')} className="input" aria-invalid={!!errors.driving_license}>
                <option value="1">Yes</option>
                <option value="0">No</option>
              </select>
              {errors.driving_license && <p className="mt-1 text-sm text-red-600">{errors.driving_license.message}</p>}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            <div>
              <label htmlFor="region_code" className="label">Region Code</label>
              <input
                id="region_code"
                type="number"
                {...register('region_code', { valueAsNumber: true })}
                className="input"
                placeholder="28"
                min={0}
                max={100}
                step={0.1}
                aria-invalid={!!errors.region_code}
              />
              {errors.region_code && <p className="mt-1 text-sm text-red-600">{errors.region_code.message}</p>}
            </div>

            <div>
              <label htmlFor="previously_insured" className="label">Previously Insured</label>
              <select id="previously_insured" {...register('previously_insured')} className="input" aria-invalid={!!errors.previously_insured}>
                <option value="0">No</option>
                <option value="1">Yes</option>
              </select>
              {errors.previously_insured && <p className="mt-1 text-sm text-red-600">{errors.previously_insured.message}</p>}
            </div>

            <div>
              <label htmlFor="vehicle_age" className="label">Vehicle Age</label>
              <select id="vehicle_age" {...register('vehicle_age')} className="input" aria-invalid={!!errors.vehicle_age}>
                <option value="< 1 Year">{"< 1 Year"}</option>
                <option value="1-2 Year">1-2 Year</option>
                <option value="> 2 Years">{"> 2 Years"}</option>
              </select>
              {errors.vehicle_age && <p className="mt-1 text-sm text-red-600">{errors.vehicle_age.message}</p>}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="vehicle_damage" className="label">Vehicle Damage History</label>
              <select id="vehicle_damage" {...register('vehicle_damage')} className="input" aria-invalid={!!errors.vehicle_damage}>
                <option value="No">No</option>
                <option value="Yes">Yes</option>
              </select>
              {errors.vehicle_damage && <p className="mt-1 text-sm text-red-600">{errors.vehicle_damage.message}</p>}
            </div>

            <div>
              <label htmlFor="annual_premium" className="label">Annual Premium ($)</label>
              <input
                id="annual_premium"
                type="number"
                {...register('annual_premium', { valueAsNumber: true })}
                className="input"
                placeholder="35000"
                min={0}
                max={1000000}
                step={100}
                aria-invalid={!!errors.annual_premium}
              />
              {errors.annual_premium && <p className="mt-1 text-sm text-red-600">{errors.annual_premium.message}</p>}
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <label htmlFor="policy_sales_channel" className="label">Policy Sales Channel</label>
              <input
                id="policy_sales_channel"
                type="number"
                {...register('policy_sales_channel', { valueAsNumber: true })}
                className="input"
                placeholder="152"
                min={0}
                max={200}
                step={0.1}
                aria-invalid={!!errors.policy_sales_channel}
              />
              {errors.policy_sales_channel && <p className="mt-1 text-sm text-red-600">{errors.policy_sales_channel.message}</p>}
            </div>

            <div>
              <label htmlFor="vintage" className="label">Vintage (Days as Customer)</label>
              <input
                id="vintage"
                type="number"
                {...register('vintage', { valueAsNumber: true })}
                className="input"
                placeholder="150"
                min={0}
                max={1000}
                aria-invalid={!!errors.vintage}
              />
              {errors.vintage && <p className="mt-1 text-sm text-red-600">{errors.vintage.message}</p>}
            </div>
          </div>

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
            <p className="mt-1 text-xs text-gray-500">Lower threshold = more predictions as "Purchase" (higher recall)</p>
          </div>

          <div className="flex items-center space-x-4 pt-4 border-t border-gray-200">
            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary"
            >
              {isLoading ? 'Predicting...' : 'Predict'}
            </button>
            <button
              type="button"
              onClick={handleClear}
              className="btn-secondary"
            >
              Clear
            </button>
          </div>
        </form>
      </div>

      {result && (
        <div className="card">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">Prediction Result</h2>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Prediction</p>
              <p className="mt-1 text-4xl font-bold text-gray-900">
                {result.prediction === 1 ? 'Will Purchase' : 'Will Not Purchase'}
              </p>
            </div>

            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Purchase Probability</p>
              <div className="mt-1">
                <ProbabilityBar probability={result.probability} />
                <p className="mt-2 text-3xl font-bold text-gray-900">
                  {(result.probability * 100).toFixed(1)}%
                </p>
              </div>
            </div>

            <div className="text-center p-4 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-500">Propensity Category</p>
              <p className="mt-1">
                <PropensityBadge category={result.propensity_category} />
              </p>
            </div>
          </div>

          <div className="mt-6 p-4 bg-gray-50 rounded-lg text-sm text-gray-600">
            <p>Threshold used: {(result.threshold_used * 100).toFixed(0)}%</p>
            {result.model_version && <p className="mt-1">Model: {result.model_version}</p>}
          </div>
        </div>
      )}
    </div>
  );
}