import React from 'react';
import './components.css';

function SecondTreatment({ name }) {
  return (
    <div className="treatment-plan">
      <h3>Advanced Care Guide for {name}</h3>
      <ul>
        <li>Medication: Dopamine agonists (Pramipexole, Ropinirole)</li>
        <li>Therapy: Yoga & Resistance Training</li>
        <li>Options: Deep brain stimulation & speech therapy</li>
      </ul>
      <h3>Recommended Foods</h3>
      <ul>
        <li>Leafy greens, Salmon, Walnuts</li>
        <li>Beans, lentils, whole wheat bread</li>
      </ul>
      <h3>To Avoid</h3>
      <ul>
        <li>Processed foods, red meats, soft drinks</li>
      </ul>
    </div>
  );
}

export default SecondTreatment;
