import requests

def hf_validate_api_token(api_token):
    
    # Define la URL de la API
    url = "https://huggingface.co/api/whoami-v2"

    # Define los encabezados de la solicitud
    headers = {
        "Authorization": f"Bearer {api_token}"
    }

    # Realiza la solicitud a la API
    response = requests.get(url, headers=headers)

    # Si la respuesta tiene un código de estado 200, el token es válido
    if response.status_code == 200:
        return True, "Welcome " + response.json()['fullname'] + "! Thank you for trying this mini-app!"
    else:
        return False, response.json()['error']