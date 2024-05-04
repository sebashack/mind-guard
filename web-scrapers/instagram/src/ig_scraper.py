import argparse
from random_username.generate import generate_username
from instagrapi import Client

parser = argparse.ArgumentParser(description='Instagram Scraper')

parser.add_argument('--hashtag', type=str, default='depression', help='Hashtag to search for')
parser.add_argument('--media_amount', type=int, default=2, help='Number of media items to retrieve')
parser.add_argument('--comment_amount', type=int, default=10, help='Number of comments per media item')
parser.add_argument('--output_file', type=str, default='result.txt', help='Output file name')

args = parser.parse_args()

username = 'USER_NAME_IG'
password = 'PASSWORD_IG'

cl = Client()
cl.login(username, password)

hashtag = args.hashtag
media_amount = args.media_amount
comment_amount = args.comment_amount

username_mappings = {}

response = cl.hashtag_medias_recent(hashtag, amount=media_amount)
with open(args.output_file, 'w') as f:
    for media in response:
        media_id = media.pk
        caption_text = media.caption_text
        comments = cl.media_comments(media_id, comment_amount)
        for comment in comments:
            content = comment.dict()
            comment_username = content['user']['username']
            text = content['text']
            if comment_username not in username_mappings:
                random_generated_username = generate_username(1)
                username_mappings[comment_username] = random_generated_username
                clean_text = text.replace("\\r\\n", "")
            f.write(f"{username_mappings[comment_username][0]}: {clean_text}\n")