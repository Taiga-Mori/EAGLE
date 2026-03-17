import streamlit.web.cli as stcli
import os
import sys



def streamlit_run():

    if hasattr(sys, "_MEIPASS"):
        working_dir = sys._MEIPASS
    else:
        working_dir = os.path.abspath(".")

    src = working_dir + '/main.py'
    sys.argv=['streamlit', 'run', src, '--global.developmentMode=false']
    sys.exit(stcli.main())

if __name__ == "__main__":
    streamlit_run()