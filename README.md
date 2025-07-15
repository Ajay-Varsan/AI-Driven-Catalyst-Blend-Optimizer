# AI-Driven-Catalyst-Blend-Optimizer
Here’s the full `README.md` content in plain **text format** (no markdown formatting) — just copy and paste this into your GitHub repo’s README:

---

🧪 AI‑Driven Catalyst Blend Optimizer

An intelligent, web-based assistant that helps chemical engineers and researchers generate optimal catalyst blends based on specific property targets — such as attrition resistance, particle size, surface area, and cost.

This tool uses SciPy’s optimization engine and a dataset of real compound properties to return the 3 best blends that match your input goals.

---

Features:

* User enters 11 catalyst property targets
* AI generates 3 optimal blend recommendations
* Blends minimize performance error and cost
* Built with Flask + JavaScript + SciPy
* Results returned instantly on the same page

---

Tech Stack:

* Python 3.12
* Flask (backend server)
* HTML + CSS + JavaScript (frontend UI)
* SciPy + NumPy + Pandas (for optimization and data handling)

---

Getting Started:

1. Clone the repository

   git clone [https://github.com/your-username/catalyst-blend-optimizer.git](https://github.com/Ajay-Varsan/catalyst-blend-optimizer.git)
   cd catalyst-blend-optimizer

2. Create a virtual environment and install dependencies

   python -m venv venv
   venv\Scripts\activate      (on Windows)
   OR
   source venv/bin/activate   (on Mac/Linux)

   pip install flask pandas numpy scipy

3. Run the app

   python app.py

Then open [http://localhost:5000](http://localhost:5000) in your browser.

---

Project Structure:

* app.py                          → Flask backend logic
* augmented\_compound\_data.csv    → Dataset used for blend calculations
* templates/index.html           → Main UI (web form)
* static/style.css               → Styling for the web app
* README.md                      → Project documentation

---

How It Works:

* Takes in 11 user-specified property values
* Builds a performance loss + cost loss function
* Uses scipy.optimize.minimize with constraints
* Generates diverse and cost-efficient blend suggestions
* Returns the top 3 ranked blends to the browser

---

