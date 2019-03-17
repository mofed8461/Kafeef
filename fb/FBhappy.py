from api import GraphAPI

graph = GraphAPI(access_token='EAAZAmEQfqYC4BAOXPwgQMZB5Xz9KeSwhQ3ksHgTlknhZChlCqk58zZCp65VqulnHzEDajBmJdjk7n8RKlE2yZBjFB8d4aZAM73Pz6lUToo451nzEZBPM2pkFoWwLGjZB9sy1xpqtfR1XSBSVSgTtvRfvgCph38ofk5ked4Y6CrtnAJKaWcEK1FnZC', version='2.9')
graph.put_object(parent_object='me', connection_name='feed',message='I\'m Happy :)')
