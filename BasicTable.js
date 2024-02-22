// Contenu de BasicTable.js
const table = `
<table>
  <thead>
    <tr>
      <th>Dessert (100g serving)</th>
      <th>Calories</th>
      <th>Fat (g)</th>
      <th>Carbs (g)</th>
      <th>Protein (g)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Frozen yoghurt</td>
      <td>159</td>
      <td>6.0</td>
      <td>24</td>
      <td>4.0</td>
    </tr>
    <!-- Autres lignes de données ici -->
  </tbody>
</table>
`;

// Afficher le tableau HTML
document.getElementById('table-container').innerHTML = table;
