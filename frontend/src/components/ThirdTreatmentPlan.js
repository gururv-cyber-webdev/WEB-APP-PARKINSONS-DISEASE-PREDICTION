import React from 'react';
import './components.css';

function ThirdTreatment({ name }) {
  return (
    <div className="treatment-plan">
      <h3>Holistic Plan for {name}</h3>
      <ul>
        <li>Medication: MAO-B inhibitors + natural remedies (on doctor’s approval)</li>
        <li>Music therapy & Art Therapy sessions</li>
        <li>Walking outdoors + Vitamin D intake</li>
      </ul>
      <h3>Daily Intake</h3>
      <ul>
        <li>Green tea, broccoli, flax seeds</li>
        <li>Curcumin-rich food (turmeric), berries</li>
      </ul>
      <h3>Avoid</h3>
      <ul>
        <li>Alcohol, caffeine in excess, fried items</li>
      </ul>
    </div>
  );
}

export default ThirdTreatment;
