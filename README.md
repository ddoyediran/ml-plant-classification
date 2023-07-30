## What will I end up with?

If you go through the steps below without fail, you should end up with a [Streamlit](http://streamlit.io/)-powered web application (Plant Disease Vision) for classifying if an apple leaf is healthy or has a disease [multiple diseases, scab, rust] (will be deployed on Google Cloud).

## 1. Getting the app running

1. Clone this repo

```
git clone https://github.com/ddoyediran/ml-plant-classification
```

2. Change into the `ml-plant-classification` directory

```
cd ml-plant-classification
```

3. Create and activate a virtual environment (call it what you want, I called mine "env")

```
pip install virtualenv
virtualenv <ENV-NAME>
source <ENV-NAME>/bin/activate
```

4. Install the required dependencies (Streamlit, TensorFlow, etc)

```
pip install -r requirements.txt
```

5. Activate Streamlit and run `app.py`

```
streamlit run app.py
```
