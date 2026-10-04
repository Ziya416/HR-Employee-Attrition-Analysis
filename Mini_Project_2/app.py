import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from pathlib import Path

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        background-color: #1e1e1e;
        color: #ffffff;
    }
    [data-testid="stSidebar"] .stRadio > label,
    [data-testid="stSidebar"] .stSelectbox > label,
    [data-testid="stSidebar"] .stTextInput > label {
        color: #ffffff;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
data_path = Path(__file__).parent / "HR_Employee_Attrition_Analysis.csv"
if not data_path.exists():
    data_path = Path(__file__).parent.parent / "HR_Employee_Attrition_Analysis.csv"

df = pd.read_csv(data_path)

model_df = df.copy()
for col in model_df.select_dtypes(include="object").columns:
    if col != "Attrition":
        model_df[col] = model_df[col].astype("category")

X = pd.get_dummies(model_df.drop(columns=["Attrition"]), drop_first=True)
y = model_df["Attrition"].map({"Yes": 1, "No": 0})

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
model_accuracy = accuracy_score(y_test, predictions)
feature_columns = X.columns

selected_page = st.sidebar.radio(
    "Select Button",
    [
        "Dashboard",
        "Dataset",
        "Visualization",
        "Performance",
        "Predict Attrition",
    ],
)

if selected_page == "Dashboard":
    st.markdown("<h1 style='text-align: center;'>📊 HR Attrition Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown(
        """
        <div style="background-color: #2b2620; border-left: 5px solid #88BDA4; border: 1px solid #4a3f30; padding: 14px 18px; border-radius: 8px; margin: 0 0 18px;">
            <p style="margin: 0 0 5px; color: #f7e1b5; font-size: 1.15rem; font-weight: 600;">
                Welcome to HR Attrition Analysis
            </p>
            <p style="margin: 0; color: #d4c5b9;">
                Explore employee trends and discover insights that can help your organization understand attrition.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("------")
    col1, col2, col3, col4 = st.columns(4)

    x = (len(df[df["Attrition"] == "Yes"]) / len(df)) * 100
    col1.metric("Total Employees", len(df))
    col2.metric("Employees Stayed", len(df[df["Attrition"] == "No"]))
    col3.metric("Employees Left", len(df[df["Attrition"] == "Yes"]))
    col4.metric("Attrition Rate", f"{x:.2f}%")

if selected_page == "Dataset":
    st.subheader("Dataset Preview")
    y = st.slider("Data", min_value=0, max_value=len(df), value=5)
    st.dataframe(df.head(y))

if selected_page == "Visualization":
    st.markdown("<h2 style='text-align: center;'>Visualization</h2>", unsafe_allow_html=True)
    st.markdown("------")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Attrition Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Attrition", ax=ax)
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
    with col2:
        st.subheader("Gender Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Gender", ax=ax)
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Department Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Department", ax=ax)
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
    with col2:
        st.subheader("Job Role Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="JobRole", ax=ax)
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=90)
        st.pyplot(fig)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Department vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="Department", hue="Attrition", ax=ax, palette="Set2")
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        dept_attrition = (df[df["Attrition"] == "Yes"].groupby("Department").size())
        highest_dept = dept_attrition.idxmax()
        highest_count = dept_attrition.max()
        st.success(f"{highest_dept} department has the highest attrition with {highest_count} employees leaving.")
        
    with col2:
        attrition_rate = (df.groupby("Department")["Attrition"].apply(lambda x: (x == "Yes").mean() * 100))
        st.subheader("Department Attrition Rate")
        st.bar_chart(attrition_rate)
        dept_high = attrition_rate.idxmax()
        dept_high_rate = attrition_rate.max()
        st.success(f"Department with highest attrition rate: {dept_high} ({dept_high_rate:.2f}%)")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Environment Satisfaction vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="EnvironmentSatisfaction", hue="Attrition", ax=ax, palette={"Yes": "red", "No": "skyblue"})
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
        env_sat = (df[df["Attrition"] == "Yes"].groupby("EnvironmentSatisfaction").size())
        env_high = env_sat.idxmax()
        env_count = env_sat.max()
        st.success(f"Employees with Environment Satisfaction level {env_high} have the highest attrition with {env_count} employees leaving.")
    with col2:
        st.subheader("Job Satisfaction vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="JobSatisfaction", hue="Attrition", ax=ax, palette={"Yes": "red", "No": "lightgreen"})
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
        job_sat = (df[df["Attrition"] == "Yes"].groupby("JobSatisfaction").size())
        job_high = job_sat.idxmax()     
        job_count = job_sat.max()
        st.success(f"Employees with Job Satisfaction level {job_high} have the highest attrition with {job_count} employees leaving.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Overtime vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="OverTime", hue="Attrition", ax=ax, palette="Set3")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
        overtime = (df[df["Attrition"] == "Yes"].groupby("OverTime").size())
        overtime_count = overtime.max()
        st.success(f"Employees who have worked overtime have the highest attrition with {overtime_count} employees leaving.")
    with col2:
        st.subheader("Work-Life Bal. vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="WorkLifeBalance", hue="Attrition", ax=ax, palette="Set1")
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
        work_life = (df[df["Attrition"] == "Yes"].groupby("WorkLifeBalance").size())
        work_life_high = work_life.idxmax()         
        work_life_count = work_life.max()
        st.success(f"Employees with Work-Life Balance level {work_life_high} have the highest attrition with {work_life_count} employees leaving.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Business Travel vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="BusinessTravel", hue="Attrition", ax=ax)
        for container in ax.containers:
            ax.bar_label(container)
        st.pyplot(fig)
        buss_tra = (df[df["Attrition"] == "Yes"].groupby("BusinessTravel").size())
        buss_tra_high = buss_tra.idxmax()
        buss_tra_count = buss_tra.max()
        st.success(f"Employees who {buss_tra_high} have the highest attrition with {buss_tra_count} employees leaving.")
    with col2:
        st.subheader("Job Role vs Attrition")
        fig, ax = plt.subplots()
        sns.countplot(data=df, x="JobRole", hue="Attrition", ax=ax, palette="coolwarm")
        for container in ax.containers:
            ax.bar_label(container)
        plt.xticks(rotation=90)
        st.pyplot(fig)
        job_role = (df[df["Attrition"] == "Yes"].groupby("JobRole").size())
        high_role = job_role.idxmax()
        high_count = job_role.max()
        st.success(f"{high_role} has the highest attrition with {high_count} employees leaving.")

    st.subheader("Monthly Income Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="MonthlyIncome", kde=True, ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)
    mon_income = (df[df["Attrition"] == "Yes"].groupby("MonthlyIncome").size())
    mon_income_high = mon_income.idxmax()
    mon_income_count = mon_income.max()
    st.success(f"Employees with Monthly Income level {mon_income_high} have the highest attrition with {mon_income_count} employees leaving.")

    st.subheader("Distance From Home Distribution")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.countplot(data=df, x="DistanceFromHome", hue="Attrition", ax=ax, palette="Set2")
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)
    dis_home = (df[df["Attrition"] == "Yes"].groupby("DistanceFromHome").size())
    dis_home_high = dis_home.idxmax()
    dis_home_count = dis_home.max() 
    st.success(f"Employees with Distance From Home {dis_home_high} have the highest attrition with {dis_home_count} employees leaving.")    

    st.subheader("Years Since Last Promotion vs Attrition")
    fig, ax = plt.subplots(figsize=(13, 8))
    sns.countplot(data=df, x="YearsSinceLastPromotion", hue="Attrition", ax=ax, palette="Set2")
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)
    years_promotion = (df[df["Attrition"] == "Yes"].groupby("YearsSinceLastPromotion").size())
    years_promotion_high = years_promotion.idxmax() 
    years_promotion_count = years_promotion.max()
    st.success(f"Employees with {years_promotion_high} years since last promotion have the highest attrition with {years_promotion_count} employees leaving.")

    st.subheader("Years With Current Manager vs Attrition")
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x="YearsWithCurrManager", hue="Attrition", ax=ax, palette="Set3")
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)
    years_manager = (df[df["Attrition"] == "Yes"].groupby("YearsWithCurrManager").size())
    years_manager_high = years_manager.idxmax()
    years_manager_count = years_manager.max()
    st.success(f"Employees with {years_manager_high} years with current manager have the highest attrition with {years_manager_count} employees leaving.")

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(data=df, x="Age", hue="Attrition", kde=True, ax=ax)
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)
    age = (df[df["Attrition"] == "Yes"].groupby("Age").size())
    age_high = age.idxmax()
    age_count = age.max()
    st.success(f"Employees with age {age_high} have the highest attrition with {age_count} employees leaving.")

    st.subheader("Years At Company vs Attrition")
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.countplot(data=df, x="YearsAtCompany", hue="Attrition", ax=ax, palette={"Yes": "red", "No": "purple"})
    for container in ax.containers:
        ax.bar_label(container)
    plt.xticks(rotation=45)
    st.pyplot(fig)
    years_at_com = (df[df["Attrition"] == "Yes"].groupby("YearsAtCompany").size())
    years_at_com_high = years_at_com.idxmax()       
    years_at_com_count = years_at_com.max()
    st.success(f"Employees with {years_at_com_high} years at company have the highest attrition with {years_at_com_count} employees leaving.")

    st.subheader("Total Working Years vs Attrition")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.countplot(data=df, x="TotalWorkingYears", hue="Attrition", ax=ax, palette="Set2")
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)
    total_working_years = (df[df["Attrition"] == "Yes"].groupby("TotalWorkingYears").size())
    total_working_years_high = total_working_years.idxmax()
    total_working_years_count = total_working_years.max()
    st.success(f"Employees with {total_working_years_high} total working years have the highest attrition with {total_working_years_count} employees leaving.")

    st.subheader("Performance Rating vs Attrition")
    fig, ax = plt.subplots(figsize=(10, 5))     
    sns.countplot(data=df, x="PerformanceRating", hue="Attrition", ax=ax, palette="Set3")
    for container in ax.containers:
        ax.bar_label(container)
    st.pyplot(fig)  
    performance_rating = (df[df["Attrition"] == "Yes"].groupby("PerformanceRating").size())
    performance_rating_high = performance_rating.idxmax()       
    performance_rating_count = performance_rating.max()
    st.success(f"Employees with Performance Rating {performance_rating_high} have the highest attrition with {performance_rating_count} employees leaving.")

    st.subheader("Correlation Analysis")
    df["Attrition"] = df["Attrition"].map({"Yes": 1, "No": 0})
    numeric_df = df.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(15, 10))
    sns.heatmap(
        numeric_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)
    attr_corr = numeric_df.corr()["Attrition"].sort_values(ascending=False)
    st.subheader("Correlation with Attrition")
    st.dataframe(attr_corr)

if selected_page == "Performance":
    st.markdown("<h2 style='text-align: center;'>Performance Analysis</h2>", unsafe_allow_html=True)
    st.markdown("-----")

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    df_ml = df.copy()
    encoder = LabelEncoder()
    for col in df_ml.select_dtypes(include="object").columns:
        df_ml[col] = encoder.fit_transform(df_ml[col])

    X = df_ml.drop("Attrition", axis=1)
    y = df_ml["Attrition"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
        
    fig, ax = plt.subplots()
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No", "Yes"],
        yticklabels=["No", "Yes"],
        ax=ax
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.subheader("Confusion Matrix")
    st.pyplot(fig)

    report = classification_report(y_test, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.subheader("Classification Report")
    st.dataframe(report_df.style.format("{:.2f}"))

    st.subheader("Model Accuracy")
    st.success(f"{accuracy * 100:.2f}%")

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    importance = pd.DataFrame({"Feature": X.columns, "Importance": model.feature_importances_}).sort_values(by="Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    st.subheader("Feature Importance")
    sns.barplot(data=importance.head(10), x="Importance", y="Feature", ax=ax)
    st.pyplot(fig)

if selected_page == "Predict Attrition":
    st.markdown("<h2 style='text-align: center;'>Predict Attrition</h2>", unsafe_allow_html=True)
    st.markdown("------")
    st.write("Fill in the employee details below to get an attrition prediction.")
    st.caption("The model uses all the listed employee features except Attrition.")

    with st.form("prediction_form"):
        col1, col2 = st.columns(2)

        with col1:
            age = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), int(df["Age"].median()))
            business_travel = st.selectbox("Business Travel", df["BusinessTravel"].unique())
            department = st.selectbox("Department", df["Department"].unique())
            distance_from_home = st.slider("Distance From Home", int(df["DistanceFromHome"].min()), int(df["DistanceFromHome"].max()), int(df["DistanceFromHome"].median()))
            education_field = st.selectbox("Education Field", sorted(df["EducationField"].unique()))
            environment_satisfaction = st.selectbox("Environment Satisfaction", sorted(df["EnvironmentSatisfaction"].unique()))
            gender = st.selectbox("Gender", df["Gender"].unique())
            hourly_rate = st.slider("Hourly Rate", min_value=int(df["HourlyRate"].min()), max_value=int(df["HourlyRate"].max()), value=int(df["HourlyRate"].mean()))
            job_involvement = st.selectbox("Job Involvement", sorted(df["JobInvolvement"].unique()))
            job_level = st.slider("Job Level", min_value=int(df["JobLevel"].min()), max_value=int(df["JobLevel"].max()), value=int(df["JobLevel"].mean()))
            job_role = st.selectbox("Job Role", df["JobRole"].unique())
            job_satisfaction = st.selectbox("Job Satisfaction", sorted(df["JobSatisfaction"].unique()))

        with col2:
            monthly_income = st.slider("Monthly Income", min_value=int(df["MonthlyIncome"].min()), max_value=int(df["MonthlyIncome"].max()), value=int(df["MonthlyIncome"].mean()))
            monthly_rate = st.slider("Monthly Rate", min_value=int(df["MonthlyRate"].min()), max_value=int(df["MonthlyRate"].max()), value=int(df["MonthlyRate"].mean()))
            num_companies_worked = st.slider("Number of Companies Worked", min_value=int(df["NumCompaniesWorked"].min()), max_value=int(df["NumCompaniesWorked"].max()), value=int(df["NumCompaniesWorked"].mean()))
            overtime = st.selectbox("Over Time", df["OverTime"].unique())
            percent_salary_hike = st.slider("Percent Salary Hike", min_value=int(df["PercentSalaryHike"].min()), max_value=int(df["PercentSalaryHike"].max()), value=int(df["PercentSalaryHike"].mean()))
            performance_rating = st.selectbox("Performance Rating", sorted(df["PerformanceRating"].unique()))
            total_working_years = st.slider("Total Working Years", min_value=int(df["TotalWorkingYears"].min()), max_value=int(df["TotalWorkingYears"].max()), value=int(df["TotalWorkingYears"].mean()))
            work_life_balance = st.selectbox("Work Life Balance", sorted(df["WorkLifeBalance"].unique()))
            years_at_company = st.slider("Years at Company", min_value=int(df["YearsAtCompany"].min()), max_value=int(df["YearsAtCompany"].max()), value=int(df["YearsAtCompany"].mean()))
            years_in_current_role = st.slider("Years in Current Role", min_value=int(df["YearsInCurrentRole"].min()), max_value=int(df["YearsInCurrentRole"].max()), value=int(df["YearsInCurrentRole"].mean()))
            years_since_last_promotion = st.slider("Years Since Last Promotion", min_value=int(df["YearsSinceLastPromotion"].min()), max_value=int(df["YearsSinceLastPromotion"].max()), value=int(df["YearsSinceLastPromotion"].mean()))
            years_with_curr_manager = st.slider("Years With Current Manager", min_value=int(df["YearsWithCurrManager"].min()), max_value=int(df["YearsWithCurrManager"].max()), value=int(df["YearsWithCurrManager"].mean()))

        submitted = st.form_submit_button("Predict Attrition")

    if submitted:
        input_data = {
            "Age": age,
            "BusinessTravel": business_travel,
            "Department": department,
            "DistanceFromHome": distance_from_home,
            "EducationField": education_field,
            "EnvironmentSatisfaction": environment_satisfaction,
            "Gender": gender,
            "HourlyRate": hourly_rate,
            "JobInvolvement": job_involvement,
            "JobLevel": job_level,
            "JobRole": job_role,
            "JobSatisfaction": job_satisfaction,
            "MonthlyIncome": monthly_income,
            "MonthlyRate": monthly_rate,
            "NumCompaniesWorked": num_companies_worked,
            "OverTime": overtime,
            "PercentSalaryHike": percent_salary_hike,
            "PerformanceRating": performance_rating,
            "TotalWorkingYears": total_working_years,
            "WorkLifeBalance": work_life_balance,
            "YearsAtCompany": years_at_company,
            "YearsInCurrentRole": years_in_current_role,
            "YearsSinceLastPromotion": years_since_last_promotion,
            "YearsWithCurrManager": years_with_curr_manager,
        }

        input_df = pd.DataFrame([input_data])
        input_df = pd.get_dummies(input_df, drop_first=True)
        input_df = input_df.reindex(columns=feature_columns, fill_value=0)

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        st.subheader("Prediction Result")
        if prediction == 1:
            st.error("⚠️ High chance this employee will leave.")
        else:
            st.success("✅ Low chance this employee will leave.")

        st.metric("Probability of Attrition", f"{probability * 100:.1f}%")
        st.progress(probability)
        st.caption(f"Model accuracy on test data: {model_accuracy * 100:.1f}%")
