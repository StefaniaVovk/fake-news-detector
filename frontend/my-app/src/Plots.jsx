import React from "react";

export default function Plots({ model, plotsReady }) {
  const apiUrl = process.env.REACT_APP_API_URL;

  if (!plotsReady) {
    return <p>⏳ Графіки ще генеруються, зачекайте...</p>;
  }

  return (
    <div className="mt-4">
      <h3 className="font-semibold mb-2">📈 Графіки моделі</h3>

      <div className="d-flex gap-4 flex-wrap mt-4">
        <div>
          <h4 className="fw-semibold">Confusion Matrix</h4>
          <img
            src={`${apiUrl}/api/ml/plots/confusion_matrix.png?model_name=${model}`}
            alt="Confusion Matrix"
            className="rounded shadow-sm"
            style={{ maxWidth: "325px" }}
          />
        </div>

        <div>
          <h4 className="fw-semibold">ROC Curve</h4>
          <img
            src={`${apiUrl}/api/ml/plots/roc_curve.svg?model_name=${model}`}
            alt="ROC Curve"
            className="rounded shadow-sm"
            style={{ maxWidth: "325px" }}
          />
        </div>
      </div>

    </div>
  );
}
