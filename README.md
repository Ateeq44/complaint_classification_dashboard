<h1>Complaint Classification Project</h1>
  <p>
    Local governments often receive large volumes of citizen complaints covering issues from potholes and water leaks to power outages.
    Efficiently categorising these complaints helps departments respond quickly and track recurring problems. This project demonstrates
    a simple text classification system for categorising complaints into common categories using a <strong>Multinomial Naive Bayes</strong> model.
    It includes a command‑line classifier and an interactive Streamlit dashboard built on a synthetic dataset.
  </p>

  <h2>Project Structure</h2>
  <table>
    <thead>
      <tr><th>File</th><th>Description</th></tr>
    </thead>
    <tbody>
      <tr>
        <td><code>complaint_classifier.py</code></td>
        <td>CLI script that trains a Naive Bayes model on complaint text and classifies user‑entered complaints.</td>
      </tr>
      <tr>
        <td><code>complaint_dashboard.py</code></td>
        <td>Streamlit dashboard enabling interactive classification, dataset summaries and sample complaints.</td>
      </tr>
      <tr>
        <td><code>complaints_data.csv</code></td>
        <td>Synthetic dataset with two columns: <code>complaint</code> (text) and <code>category</code> (e.g. Road, Water, Electricity, Waste, Health, Other).</td>
      </tr>
    </tbody>
  </table>

  <h2>Features</h2>
  <h3>CLI Script</h3>
  <ul>
    <li><strong>Training and evaluation</strong> – Reads <code>complaints_data.csv</code>, splits it 80/20 into training and test sets, converts complaints into a bag‑of‑words representation (<code>CountVectorizer</code> with English stop‑word removal), trains a <strong>Multinomial Naive Bayes</strong> classifier and prints the accuracy and classification report on the test set.</li>
    <li><strong>Interactive classification</strong> – After training, prompts you to enter complaint text and returns the predicted category until you type <code>quit</code>.</li>
  </ul>
  <p>Run the CLI with:</p>
  <pre><code>python complaint_classifier.py</code></pre>

  <h3>Streamlit Dashboard</h3>
  <p>The dashboard offers a user‑friendly interface to explore and classify complaints:</p>
  <ul>
    <li><strong>Complaint classifier</strong> – Enter a complaint description in the text area to see the predicted category immediately.</li>
    <li><strong>Dataset summary</strong> – Displays the number of complaints in each category in a table.</li>
    <li><strong>Sample complaints</strong> – Provides example complaints for each category organised into tabs to help understand the data.</li>
    <li><strong>Model accuracy</strong> – Shows the accuracy of the trained classifier on the test set at runtime.</li>
  </ul>
  <p>Launch the dashboard with:</p>
  <pre><code>python -m streamlit run complaint_dashboard.py</code></pre>

  <h2>Installation</h2>
  <ol>
    <li>Ensure Python 3.8+ is installed on your system.</li>
    <li>Install required packages:
      <pre><code>python -m pip install streamlit pandas scikit-learn</code></pre>
    </li>
  </ol>

  <h2>Customising the Dataset</h2>
  <p>To adapt the model to your own complaint data:</p>
  <ol>
    <li>Create a CSV file with at least two columns:
      <ul>
        <li><code>complaint</code> – Free‑text description of the issue.</li>
        <li><code>category</code> – Label representing the category (e.g. Road, Water, Electricity, Waste, Health, Other). You can define your own categories.</li>
      </ul>
    </li>
    <li>Save the file as <code>complaints_data.csv</code> in the same directory as the scripts or adjust the code to read your filename.</li>
    <li>The classifier currently uses simple bag‑of‑words features. For longer messages or more nuanced classification, consider using TF‑IDF features, n‑grams or modern embeddings like word2vec or BERT.</li>
  </ol>

  <h2>Model Considerations</h2>
  <p>The Naive Bayes algorithm assumes independence between words and works best with relatively short text messages. For improved accuracy, you could experiment with logistic regression, support vector machines or transformer‑based models, particularly when training on larger, more diverse datasets.</p>
