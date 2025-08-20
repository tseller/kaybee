from floggit import flog

@flog
def get_prompt():
    return '''You are an AI assistant who is a fountain of knowledge about {company}.'''
