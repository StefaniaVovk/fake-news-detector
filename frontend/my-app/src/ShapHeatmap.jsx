// src/ShapHeatmap.jsx
import React, { useState } from "react";
import { Bar } from "react-chartjs-2";
import {
    Chart as ChartJS,
    CategoryScale,
    LinearScale,
    BarElement,
    Title,
    Tooltip,
    Legend
} from "chart.js";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);


export default function ShapHeatmap({ payload, text, prediction }) {
    const [selectedIndex, setSelectedIndex] = useState(null);

    console.log("🔍 prediction:", prediction);

    if (!payload || !Array.isArray(payload)) {
        console.warn("⚠️ Немає SHAP значень або невірний формат:", payload);
        return null;
    }

    // --- Отримуємо значення SHAP (scores) ---
    let values = payload;
    while (Array.isArray(values[0])) values = values[0];

    const tokens = text.split(/\s+/);
    if (tokens.length !== values.length) {
        console.warn(`⚠️ Токенів ${tokens.length}, а SHAP значень ${values.length}`);
    }

    // --- Масштабування ---
    const maxAbs = Math.max(...values.map(v => Math.abs(v)), 1e-6);

    // --- Функція кольору ---
    const getColor = (val) => {
        const norm = val / maxAbs;
        if (norm > 0) {
            const intensity = Math.min(1, norm);
            return `rgba(255, 0, 0, ${intensity * 0.6})`; // червоні — підвищують
        } else {
            const intensity = Math.min(1, -norm);
            return `rgba(0, 0, 255, ${intensity * 0.6})`; // сині — знижують
        }
    };

    // --- Клік по слову ---
    const handleTokenClick = (i) => {
        setSelectedIndex(selectedIndex === i ? null : i);
    };

    const combined = tokens.map((t, i) => ({ token: t, value: values[i] }));
    const top5 = combined
        .sort((a, b) => Math.abs(b.value) - Math.abs(a.value))
        .slice(0, 5);

    const topTokens = top5.map(d => d.token);
    const topValues = top5.map(d => d.value);

    return (
        <div className="p-3 border rounded bg-white shadow-sm leading-relaxed text-lg text-justify">
            <h3 className="font-semibold mb-3">SHAP пояснення впливу токенів</h3>

            <p
                style={{
                    lineHeight: "1.8em",
                    maxWidth: "100%",         // не ширше контейнера
                    wordWrap: "break-word",   // переносить довгі слова
                    overflowWrap: "break-word",
                    display: "flex",
                    flexWrap: "wrap",         // переносить токени вниз
                    gap: "4px",
                }}
            >
                {tokens.map((token, i) => (
                    <span
                        key={i}
                        onClick={() => handleTokenClick(i)}
                        title={`SHAP: ${values[i]?.toFixed(4)}`}
                        style={{
                            backgroundColor: getColor(values[i] || 0),
                            borderRadius: "3px",
                            marginRight: "3px",
                            padding: "2px 4px",
                            cursor: "pointer",
                            transition: "background-color 0.2s ease",
                        }}
                    >
                        {token}
                    </span>
                ))}
            </p>

            {selectedIndex !== null && (
                <div className="mt-3 text-sm text-gray-700">
                    <b>Виділене слово:</b> <code>{tokens[selectedIndex]}</code><br />
                    <b>Вплив (SHAP):</b> {values[selectedIndex].toFixed(5)}
                </div>
            )}

            <div className="mt-4 text-sm text-gray-600 italic">
                🔴 Червоні області підвищують вихід моделі,
                🔵 Сині — знижують.
                Натисни на слово, щоб побачити його вплив.
            </div>


            <div className="mt-6">
                <Bar
                    data={{
                        labels: topTokens,
                        datasets: [
                            {
                                label: "Top-5 SHAP Attribution",
                                data: topValues,
                                backgroundColor: "rgba(255, 165, 0, 0.8)", // помаранчеві стовпчики
                                borderRadius: 4,
                            },
                        ],
                    }}
                    options={{
                        responsive: true,
                        plugins: {
                            legend: { display: false },
                            title: {
                                display: true,
                                text: `SHAP-semantic attributes — ${prediction?.predicted_label?.charAt(0).toUpperCase() + prediction?.predicted_label?.slice(1)
                                    }`,
                                font: { size: 14 },
                            },
                        },
                        scales: {
                            x: {
                                ticks: {
                                    autoSkip: false,
                                    maxRotation: 60,
                                    minRotation: 45,
                                },
                            },
                            y: {
                                title: {
                                    display: true,
                                    text: "Attribution (a.u.)",
                                },
                                beginAtZero: true,
                            },
                        },
                    }}
                />
            </div>

        </div>
    );
}
