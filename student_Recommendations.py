import streamlit as st
import streamlit.components.v1 as stc

import pandas as pd
import neattext.functions as nfx

import random

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
        'company', 'similarity_score', 'profile', 'Location', 'Stipend', 'Skills and Perks', "_id"]]
    return final_recommended_internships.head(num_of_rec), idx


@st.cache
def get_inter_internship_recommendation(id, df):

    internship_indices = pd.Series(
        df.index, index=df['_id'])

    idx = internship_indices[id]

    df['clean_Skills and Perks'] = df['Skills and Perks'].apply(
        nfx.remove_stopwords)
    df['clean_Skills and Perks'] = df['clean_Skills and Perks'].apply(
        nfx.remove_special_characters)
    count_vect = CountVectorizer()
    cv_mat = count_vect.fit_transform(df['clean_Skills and Perks'])

    cosine_sim_mat = cosine_similarity(cv_mat)

    sim_scores = list(enumerate(cosine_sim_mat[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    selected_internship_indices = [i[0] for i in sim_scores[1:]]
    selected_internship_scores = [i[1] for i in sim_scores[1:]]

    result_df = df.iloc[selected_internship_indices]
    result_df['similarity_score'] = selected_internship_scores
    final_recommended_internships = result_df[[
        'company', 'similarity_score', 'profile', 'Location', 'Stipend', 'Skills and Perks']]
    return final_recommended_internships.head(4)


RESULT_TEMP = """
<div style="width:90%;height:100%;margin:1px;padding:5px;position:relative;border-radius:5px;border-radius: 40px;
box-shadow:5px 5px 4px 4px #ccc; background-color: gray;
  border-left: 5px solid #6c6c6c;">
<h4 style="color:black;font-weight:bold;margin:20px;">{}</h4>
<p style="color:black;font-weight:bold;margin:20px;"><span style="color:#ccc;font-weight:bold;">Score::</span>{}</p>
<p style="color:black;font-weight:bold;margin:20px;"><span style="color:#ccc;font-weight:bold;">Link::</span>{}</p>
<p style="color:black;font-weight:bold;margin:20px;"><span style="color:#ccc;font-weight:bold;">Price::</span>{}</p>
<p style="color:black;font-weight:bold;margin:20px;"><span style="color:#ccc;font-weight:bold;">👨🏽‍🎓 Students::</span>{}</p>

</div>
"""

RESULT_TEMP1 = """
<div style="width:90%;height:100%;margin:1px;margin-top:10px;padding:5px;position:relative;border-radius:40px;
box-shadow:5px 5px 4px 4px #ccc; background-color: gray;">
<h4 style="color:black;font-weight:bold;margin:20px;">{}</h4>
<p style="color:black;font-weight:bold;margin:20px;"><span style="color:#ccc;font-weight:bold;">score::</span>{}<span style="color:#ccc;margin-left:30px;">profile::</span>{}</p>
<p style="color:black;font-weight:bold;margin:20px;"><span style="color:#ccc;font-weight:bold;">location::</span>{}<span style="color:#ccc;margin-left:115px;">Stipend::</span>{}</p>
<p style="color:black;font-weight:bold;margin:20px;"><span style="color:#ccc;font-weight:bold;">👨🏽‍🎓 Skills and Perks:</span>{}</p>

</div>
"""

RESULT_TEMP2 = """
<div style="width:80%;height:100%;margin:1px;margin-top:10px;padding:2vh;position:relative;border-radius:40px;
box-shadow:3px 3px 3px 3px #DC143C; background-color: #ccc;">
<h4 style="color:black;font-weight:bold;margin:20px;">{}</h4>
<p style="color:#DC143C;font-weight:bold;margin:20px;"><span style="color:black;font-weight:bold;">score::</span>{}<span style="color:black;margin-left:30px;">profile::</span>{}</p>
<p style="color:#DC143C;font-weight:bold;margin:20px;"><span style="color:black;font-weight:bold;">location::</span>{}<span style="color:black;margin-left:75px;">Stipend::</span>{}</p>
<p style="color:#DC143C;font-weight:bold;margin:20px;"><span style="color:black;font-weight:bold;">👨🏽‍🎓 Skills and Perks:</span>{}</p>

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

d1 = load_data("HYDERABAD.csv")
d2 = load_data("MUMBAI.csv")
d3 = load_data("BANGALORE.csv")
d4 = load_data("DELHI.csv")
d5 = load_data("KOLKATA.csv")
d6 = load_data("INTERNATIONAL.csv")

d7 = d1.append(d2, ignore_index=True)
d8 = d7.append(d3, ignore_index=True)
d9 = d8.append(d4, ignore_index=True)
d10 = d9.append(d5, ignore_index=True)
df2 = d10.append(d6, ignore_index=True)

def main():

    st.title("student Recommendation App")

    menu = ["Home", "Courses", "Internships", "About"]
    choice = st.sidebar.selectbox("Menu", menu)

    df1 = load_data("udemy_courses.csv")
    df8 = df2

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

                    for row in results.iterrows():
                        rec_title = row[1][0]
                        rec_score = row[1][1]
                        rec_url = row[1][2]
                        rec_price = row[1][3]
                        rec_num_sub = row[1][4]

                        stc.html(RESULT_TEMP.format(rec_title, rec_score,
                                 rec_url, rec_price, rec_num_sub), height=300)
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
                    results, idx = get_internship_recommendation(
                        search_term, cosine_sim_mat, df6, num_of_rec)

                    for row in results.iterrows():
                        rec_company = row[1][0]
                        rec_profile = row[1][1]
                        rec_score = row[1][2]
                        rec_location = row[1][3]
                        rec_stipend = row[1][4]
                        rec_skills = row[1][5]
                        rec_id = row[1][6]

                        stc.html(RESULT_TEMP1.format(rec_company, rec_profile,
                                 rec_score, rec_location, rec_stipend, rec_skills), height=250)

                        with st.expander("Show Similar Internships"):
                            result = get_inter_internship_recommendation(
                                rec_id, df8)
                            for row in result.iterrows():
                                rec_company1 = row[1][0]
                                rec_profile1 = row[1][1]
                                rec_score1 = row[1][2]
                                rec_location1 = row[1][3]
                                rec_stipend1 = row[1][4]
                                rec_skills1 = row[1][5]
                                stc.html(RESULT_TEMP2.format(rec_company1, rec_profile1,
                                                             rec_score1, rec_location1, rec_stipend1, rec_skills1), height=250)

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
