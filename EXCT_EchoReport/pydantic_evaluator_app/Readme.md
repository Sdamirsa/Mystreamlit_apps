# Instruction:
Create a virtual machine
```
python -m venv venv
```
activate it (in linux and mac it's like this:)
```
source venv/bin/activate
```
Install libraires
```
pip install -r requirements.txt
```
Run the app
```
streamlit run app.py
```
Then you should see something like this:
```
(venv) (base) amir@As-MacBook-Pro pydantic_evaluator_app % streamlit run app.py 
  You can now view your Streamlit app in your browser.
  Local URL: http://localhost:8502
  Network URL: http://169.254.19.0:8502
```
You can now see the UI at http://localhost:8502