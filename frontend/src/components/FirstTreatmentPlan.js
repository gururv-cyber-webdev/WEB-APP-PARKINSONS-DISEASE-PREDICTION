import React from 'react';
import './components.css';

function FirstTreatment({ name }) {
  return (
    <div className="treatment-plan">
      <h3>Personalized Plan for {name}</h3>
      <ul>
        <li>Medication: Levodopa + Carbidopa</li>
        <li>Therapy: Tai Chi, Physiotherapy, and voice training</li>
        <li>Advanced Care: Consider Deep Brain Stimulation for severe symptoms</li>
      </ul>
      <h3>Diet Focus</h3>
      <ul>
        <li>Spinach, tofu, beef (Iron)</li>
        <li>Oats, whole grains, lean chicken (Zinc)</li>
      </ul>
      <h3>Avoid</h3>
      <ul>
        <li>Saturated fats, sugary drinks, smoking</li>
      </ul>
    </div>
  );
}

export default FirstTreatment;
