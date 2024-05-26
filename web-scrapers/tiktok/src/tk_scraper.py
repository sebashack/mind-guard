import sys
import argparse
from TikTokApi import TikTokApi
import asyncio
import os
import time
from random_username.generate import generate_username
import csv

ms_token = "xdJYcYuYsjoeJi1uNut0FgD9hmhk7K1ZPcIkd00glLoPosrBaA1dRx0NZjjNZRNju9-kRJuWn-r6FQfasifunflb-AeSk_yE86hG4c8YEGhMpjx3LBnKzq5WzsZ-9lwukTcWyIHH2MWOXDw43Yu7YWCR9A=="

def get_arguments():
    parser = argparse.ArgumentParser(description='TikTok Scraper')

    parser.add_argument('--hashtag', type=str, default='depression', help='Hashtag to search for')
    parser.add_argument('--media_amount', type=int, default=2, help='Number of media items to retrieve')
    parser.add_argument('--comment_amount', type=int, default=10, help='Number of comments per media item')
    parser.add_argument('--output_file', type=str, default='output-tk.tsv', help='Output file name')
    
    return parser.parse_args()

async def get_comments(hashtag_name, media_amount, comment_amount, output_file):
    username_mappings = {}
    user_id_counter = 1
    total_comments_written = 0

    async with TikTokApi() as api:
        await api.create_sessions(
            ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False
        )
        hashtag = api.hashtag(name=hashtag_name)
        with open(output_file, 'w', newline='', encoding='utf-8') as file:
            tsv_writer = csv.writer(file, delimiter='\t')
            tsv_writer.writerow(['anonymous_nickname', 'comment', 'id'])
            comment_amount = media_amount * comment_amount
            async for video in hashtag.videos(count=media_amount):
                if total_comments_written >= comment_amount:
                    break
                print(f"##### video-id: {video.id}:")
                async for comment in video.comments(count=comment_amount):
                    if total_comments_written >= comment_amount:
                        break
                    actual_username = comment.as_dict['user']['unique_id']
                    text = comment.as_dict['text'].replace("\n", "").replace("\t", "")
                    if actual_username not in username_mappings:
                        random_generated_username = generate_username(1)[0]
                        username_mappings[actual_username] = (random_generated_username, user_id_counter)
                        user_id_counter += 1
                    random_username, user_uuid = username_mappings[actual_username]
                    print(f"{random_username}\t{text}\t{user_uuid}")
                    tsv_writer.writerow([random_username, text, user_uuid])
                    total_comments_written += 1

if __name__ == "__main__":
    args = get_arguments()
    asyncio.run(get_comments(args.hashtag, args.media_amount, args.comment_amount, args.output_file))
