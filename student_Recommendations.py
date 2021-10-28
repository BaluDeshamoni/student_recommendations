import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
import neattext.functions as nfx

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def load_data(data):
    df = pd.read_csv(data)
    return df


def vectorize_text_to_cosine_mat(data):
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(data)
    cosine_sim_mat = cosine_similarity(cv_mat)
    return cosine_sim_mat


@st.cache
def get_course_recommendation(title, cosine_sim_mat, df, num_of_rec=10):

    course_indices = pd.Series(
        df.index, index=df['clean_course_title']).drop_duplicates()

    idx = course_indices[title]

    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_course_indices = [i[0] for i in sim_scores[1:]]
    selected_course_scores = [i[1] for i in sim_scores[1:]]

    result_df = df.iloc[selected_course_indices]
    result_df['similarity_score'] = selected_course_scores
    final_recommended_courses = result_df[[
        'course_title', 'similarity_score', 'url', 'price', 'num_subscribers']]
    return final_recommended_courses.head(num_of_rec)


@st.cache
def get_internship_recommendation(profile, cosine_sim_mat, df, num_of_rec=10):

    internship_indices = pd.Series(
        df.index, index=df['clean_profile']).drop_duplicates()

    idx = internship_indices[profile]

    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    selected_internship_indices = [i[0] for i in sim_scores[1:]]
    selected_internship_scores = [i[1] for i in sim_scores[1:]]

    result_df = df.iloc[selected_internship_indices]
    result_df['similarity_score'] = selected_internship_scores
    final_recommended_internships = result_df[[
        'company', 'similarity_score', 'profile', 'Location', 'Stipend', 'Skills and Perks']]
    return final_recommended_internships.head(num_of_rec)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">🔗</span><a href="{}",target="_blank">Link</a></p>
<p style="color:blue;"><span style="color:black;">💲Price:</span>{}</p>
<p style="color:blue;"><span style="color:black;">🧑‍🎓👨🏽‍🎓 Students:</span>{}</p>

</div>
"""

RESULT_TEMP1 = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-bottom-right-radius: 60px;
box-shadow:0 0 15px 5px #ccc; background-color: #a8f0c6;
  border-left: 5px solid #6c6c6c;">
<h4>{}</h4>
<p style="color:blue;"><span style="color:black;">📈Score::</span>{}</p>
<p style="color:blue;"><span style="color:black;">profile::</span>{}<span style="color:black;margin:20px;">location::</span>{}</p>
<p style="color:blue;"><span style="color:black;">💲Stipend:</span>{}</p>
<p style="color:blue;"><span style="color:black;">👨🏽‍🎓 Skills and Perks:</span>{}</p>

</div>
"""


@st.cache
def search_term_if_not_found(term, df):
    result_df = df[df['course_title'].str.contains(term)]
    return result_df


@st.cache
def search_term_if_not_found1(term, df):
    result_df = df[df['profile'].str.contains(term)]
    return result_df


def main():

    st.title("student Recommendation App")

    menu = ["Home", "Courses", "Internships", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    df1 = load_data("udemy_courses.csv")
    df2 = load_data("HYDERABAD.csv")

    if choice == "Home":
        st.subheader("Home")
        st.write("courses data")
        st.dataframe(df1.head(10))

        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(df1)
        st.download_button(
            label="Download above data as CSV",
            data=csv,
            file_name='udemy_courses.csv',
            mime='text/csv',
        )

        st.write("internships data")
        st.dataframe(df2.head(10))

        @st.cache
        def convert_df(df):
            return df.to_csv().encode('utf-8')
        csv = convert_df(df2)
        st.download_button(
            label="Download above data as CSV",
            data=csv,
            file_name='internships.csv',
            mime='text/csv',
        )

        st.write("Suggestions here!")
        uploaded_file = st.file_uploader("Choose a file")
        if uploaded_file is not None:
            dataframe = pd.read_csv(uploaded_file)
            st.write(dataframe)

    elif choice == "Courses":
        st.subheader("Courses")

        search_term = st.text_input("Search")

        df3 = pd.DataFrame([{'course_title': search_term}])
        df4 = df3.append(df1, ignore_index=True)

        df4['clean_course_title'] = df4['course_title'].apply(
            nfx.remove_stopwords)
        df4['clean_course_title'] = df4['clean_course_title'].apply(
            nfx.remove_special_characters)

        cosine_sim_mat = vectorize_text_to_cosine_mat(
            df4['clean_course_title'])

        num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_course_recommendation(
                        search_term, cosine_sim_mat, df4, num_of_rec)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_score = row[1][1]
                        rec_url = row[1][2][1:]
                        rec_price = row[1][3]
                        rec_num_sub = row[1][4]

                        stc.html(RESULT_TEMP.format(rec_title, rec_score,
                                 rec_url, rec_price, rec_num_sub), height=350)
                except:
                    results = "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found(search_term, df1)
                    st.dataframe(result_df)

    elif choice == "Internships":
        st.subheader("Internships")

        search_term = st.text_input("Search Bar")

        df2['clean_profile'] = df2['profile'].apply(nfx.remove_stopwords)
        df2['clean_profile'] = df2['clean_profile'].apply(
            nfx.remove_special_characters)

        df5 = pd.DataFrame(
            [{'clean_profile': search_term, 'profile': search_term}])
        df6 = df5.append(df2, ignore_index=True)

        cosine_sim_mat = vectorize_text_to_cosine_mat(
            df6['clean_profile'])

        num_of_rec = st.sidebar.number_input("Number", 4, 30, 7)
        if st.button("Recommend"):
            if search_term is not None:
                try:
                    results = get_internship_recommendation(
                        search_term, cosine_sim_mat, df6, num_of_rec)
                    with st.expander("Results as JSON"):
                        results_json = results.to_dict('index')
                        st.write(results_json)

                    for row in results.iterrows():
                        rec_company = row[1][0]
                        rec_profile = row[1][1]
                        rec_score = row[1][2]
                        rec_location = row[1][3]
                        rec_stipend = row[1][4]
                        rec_skills = row[1][5]

                        stc.html(RESULT_TEMP1.format(rec_company, rec_profile,
                                 rec_score, rec_location, rec_stipend, rec_skills), height=350)
                except:
                    results = "Not Found"
                    st.warning(results)
                    st.info("Suggested Options include")
                    result_df = search_term_if_not_found1(search_term, df2)
                    st.dataframe(result_df)

    else:
        st.subheader("About")
        st.text(
            " This is an application which recommends courses and internships for students.")
        st.text(" Recommendations are given based on the users search.")


if __name__ == '__main__':
    main()
