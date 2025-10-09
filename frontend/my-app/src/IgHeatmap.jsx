import React, { useState } from "react";

export default function IgHeatmap({ payload, text }) {
    const [selectedIndex, setSelectedIndex] = useState(null);

    if (!payload || !payload.scores || !Array.isArray(payload.scores)) {
        console.warn("⚠️ Немає IG значень або невірний формат:", payload);
        return null;
    }

    let values = payload.scores;
    while (Array.isArray(values[0])) values = values[0];

    const tokens = payload.tokens || text.split(/\s+/);
    if (tokens.length !== values.length) {
        console.warn(`⚠️ Токенів ${tokens.length}, а IG значень ${values.length}`);
    }

    // --- Масштабування ---
    const maxAbs = Math.max(...values.map(v => Math.abs(v)), 1e-6);

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

    const handleTokenClick = (i) => {
        setSelectedIndex(selectedIndex === i ? null : i);
    };

    return (
        <div className="p-3 border rounded bg-white shadow-sm leading-relaxed text-lg text-justify">
            <h3 className="font-semibold mb-3">IG пояснення впливу токенів</h3>

            <p
                style={{
                    lineHeight: "1.8em",
                    maxWidth: "100%",
                    wordWrap: "break-word",
                    overflowWrap: "break-word",
                    display: "flex",
                    flexWrap: "wrap",
                    gap: "4px",
                }}
            >
                {tokens.map((token, i) => (
                    <span
                        key={i}
                        onClick={() => handleTokenClick(i)}
                        title={`IG: ${values[i]?.toFixed(4)}`}
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
                    <b>Вплив (IG):</b> {values[selectedIndex].toFixed(5)}
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
