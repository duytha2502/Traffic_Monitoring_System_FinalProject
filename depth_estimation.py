import streamlit as st
import streamlit_authenticator as stauth
import yaml
from yaml.loader import SafeLoader

with open('config.yaml') as file:
    config = yaml.load(file, Loader=SafeLoader)

# Pre-hashing all plain text passwords once
# stauth.Hasher.hash_passwords(config['credentials'])

authenticator = stauth.Authenticate(
    config['credentials'],
    config['cookie']['name'],
    config['cookie']['key'],
    config['cookie']['expiry_days']
)

if st.session_state['authentication_status']:
    authenticator.logout()
    st.write(f'Welcome *{st.session_state["name"]}*')
    st.title('Some content')
elif st.session_state['authentication_status'] is False:
    st.error('Username/password is incorrect')
elif st.session_state['authentication_status'] is None:
    st.warning('Please enter your username and password')

tab1, tab2, tab3 = st.tabs(["Sign In", "Sign Up", "Forgot Password"])

with tab1:
    try:
        authenticator.login()
    except Exception as e:
        st.error(e)
    try:
        authenticator.experimental_guest_login('Login with Google',
                                            provider='google',
                                            oauth2=config['oauth2'])
    except Exception as e:
        st.error(e)
with tab2:
    try:
        email_of_registered_user, \
        username_of_registered_user, \
        name_of_registered_user = authenticator.register_user(merge_username_email=True)
        if email_of_registered_user:
            st.success('User registered successfully')
    except Exception as e:
        st.error(e)
with tab3:
    try:
        username_of_forgotten_password, \
        email_of_forgotten_password, \
        new_random_password = authenticator.forgot_password()
        if username_of_forgotten_password:
            st.write(new_random_password)
            st.success('New password to be sent securely')
            # The developer should securely transfer the new password to the user.
        elif username_of_forgotten_password == False:
            st.error('Username not found')
    except Exception as e:
        st.error(e)

with open('config.yaml', 'w') as file:
    yaml.dump(config, file, default_flow_style=False)