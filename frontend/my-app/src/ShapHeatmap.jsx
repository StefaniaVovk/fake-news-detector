// src/ShapHeatmap.jsx
import React, { useState } from "react";

export default function ShapHeatmap({ payload, text }) {
    // 🔹 1. Викликаємо useState одразу — до будь-яких умов
    const [selectedIndex, setSelectedIndex] = useState(null);

    // 🔹 2. Перевіряємо payload вже після
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
        </div>
    );
}
