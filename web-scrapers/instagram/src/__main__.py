import sys
import argparse
from instagrapi import Client
from random_username.generate import generate_username
import csv

username_mappings = {}
user_id_counter = 1

def create_client(username, password):
    cl = Client()
    cl.login(username, password)

    return cl

def get_arguments():
    parser = argparse.ArgumentParser(description='Instagram Scraper')

    parser.add_argument('--hashtag', type=str, default='depression', help='Hashtag to search for')
    parser.add_argument('--media_amount', type=int, default=2, help='Number of media items to retrieve')
    parser.add_argument('--comment_amount', type=int, default=10, help='Number of comments per media item')
    parser.add_argument('--output_file', type=str, default='result.txt', help='Output file name')
    parser.add_argument('--username', type=str, help='Username instagram')
    parser.add_argument('--password', type=str, help='Password instagram')
    
    return parser.parse_args()

def scraper(args, cl):
    response = cl.hashtag_medias_recent(args.hashtag, amount=args.media_amount)
    with open(args.output_file, 'w', newline='') as file:
        tsv_writer = csv.writer(file, delimiter='\t')
        tsv_writer.writerow(['anonymous_nickname', 'comment', 'id'])

        for media in response:
            media_id = media.pk
            caption_text = media.caption_text
            comments = cl.media_comments(media_id, args.comment_amount)
            for comment in comments:
                content = comment.dict()
                actual_username = content['user']['username']
                text = content['text'].replace("\n", "").replace("\t", "")
                if actual_username not in username_mappings:
                    random_generated_username = generate_username(1)[0]
                    username_mappings[actual_username] = (random_generated_username, user_id_counter)
                    user_id_counter += 1
                
                random_username, user_uuid = username_mappings[actual_username]
                print(f"{random_username}\t{text}\t{user_uuid}")
                tsv_writer.writerow([random_username, text, user_uuid])

def main():
    args = get_arguments()
    client = create_client(args.username, args.password)
    scraper(args, client)

if __name__ == "__main__":
    sys.exit(main())