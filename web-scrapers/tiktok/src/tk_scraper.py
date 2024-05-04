from TikTokApi import TikTokApi
import asyncio
import os

ms_token = "xxxxxxxxxxxxxxxxxxx"


async def get_comments():
    async with TikTokApi() as api:
        await api.create_sessions(
            ms_tokens=[ms_token], num_sessions=1, sleep_after=3, headless=False
        )
        hashtag = api.hashtag(name="depression")
        async for video in hashtag.videos():
            print(f"##### video-id: {video.id}:")
            async for comment in video.comments(count=2):
                print(comment.as_dict["text"])


if __name__ == "__main__":
    asyncio.run(get_comments())