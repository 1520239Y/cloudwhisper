import asyncio
import io
import os
import wave
import numpy as np
import sounddevice as sd
from tempfile import NamedTemporaryFile
import time  # 時間計測用

# OpenAI
from openai import AsyncOpenAI

# langchain / langgraph 関連
from typing import Literal
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode

# pydub で音声を再生
from pydub import AudioSegment
from pydub.playback import play

# ==== Unitreeロボット用の例 (不要ならコメントアウト) ====
try:
    from actions import Go2Action
    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    ROBOT_AVAILABLE = True
except ImportError as e:
    print(f"[WARN] Unitree SDK (or actions.py) Import Error: {e}")
    print("[WARN] ロボット機能は無効化されます。")
    ROBOT_AVAILABLE = False
    Go2Action = None

# ==================================================
# グローバル変数や設定値
# ==================================================
is_tts_playing = False

# ---- 無音判定用パラメータ ----
SILENCE_THRESHOLD = 700  # この値以下なら「無音」とみなす(振幅)
MAX_SILENCE_BLOCKS = 2   # 連続で何ブロック「無音」が続いたら話し終わりと判断
current_buffer = bytearray()
silence_count = 0

# -----------------------------
#  処理時間計測用変数
# -----------------------------
current_timing = {
    "capture_start": None,
    "capture_end": None,
    "transcription_start": None,
    "transcription_end": None,
    "LLM_start": None,
    "LLM_end": None,
    "TTS_start": None,
    "TTS_end": None,
    # ロボット行動系
    "robot_action_start": None,
    "robot_action_end": None,
}

def print_timing_info():
    """current_timing に格納された時間差分をログ出力"""
    # 1) 音声取得
    if current_timing["capture_start"] and current_timing["capture_end"]:
        capture_time = current_timing["capture_end"] - current_timing["capture_start"]
        print(f"[Time] 音声取得: {capture_time:.3f} 秒")

    # 2) テキスト変換 (Whisper)
    if current_timing["transcription_start"] and current_timing["transcription_end"]:
        trans_time = current_timing["transcription_end"] - current_timing["transcription_start"]
        print(f"[Time] テキスト変換: {trans_time:.3f} 秒")

    # 3) テキスト→LLM(解析)
    if current_timing["LLM_start"] and current_timing["LLM_end"]:
        llm_time = current_timing["LLM_end"] - current_timing["LLM_start"]
        print(f"[Time] LLM解析: {llm_time:.3f} 秒")

    # 4) 応答をTTS化
    if current_timing["TTS_start"] and current_timing["TTS_end"]:
        tts_time = current_timing["TTS_end"] - current_timing["TTS_start"]
        print(f"[Time] TTS生成: {tts_time:.3f} 秒")

    # 5) 犬型ロボットに行動させる (ツール呼び出し)
    if current_timing["robot_action_start"] and current_timing["robot_action_end"]:
        robot_time = current_timing["robot_action_end"] - current_timing["robot_action_start"]
        print(f"[Time] ロボット行動: {robot_time:.3f} 秒")

    print("------")

def reset_timing_info():
    """次回計測に備えて current_timing をリセット"""
    for key in current_timing.keys():
        current_timing[key] = None

# -----------------------------
#  1) ロボットアクション (ツール定義)
# -----------------------------
if ROBOT_AVAILABLE:
    try:
        # 存在しないインターフェイス名だとここでエラーが起きるので try-except でガード
        ChannelFactoryInitialize(0, "enp0s25")
        action = Go2Action()
        print("[INFO] ロボットSDK初期化成功")
    except Exception as e:
        print(f"[WARN] ロボットSDK初期化失敗: {e}")
        print("[WARN] ロボット機能は無効化されます。")
        ROBOT_AVAILABLE = False
        action = None
else:
    action = None

@tool
def StandUp():
    """立ち上がる"""
    if action:
        global current_timing
        current_timing["robot_action_start"] = time.time()
        action.StandUp()
        current_timing["robot_action_end"] = time.time()
    else:
        print("[INFO] Robot action is unavailable (StandUp).")

@tool
def SitDown():
    """座る"""
    if action:
        global current_timing
        current_timing["robot_action_start"] = time.time()
        action.SitDown()
        current_timing["robot_action_end"] = time.time()
    else:
        print("[INFO] Robot action is unavailable (SitDown).")

@tool
def Move(x: float, y: float, z: float):
    """
    前方にx(m)、右にy(m)移動し、z(rad)回転する。
    """
    if action:
        global current_timing
        current_timing["robot_action_start"] = time.time()
        action.Move(x, y, z)
        current_timing["robot_action_end"] = time.time()
    else:
        print("[INFO] Robot action is unavailable (Move).")

@tool
def Stretch():
    """ストレッチ"""
    if action:
        action.Stretch()
    else:
        print("[INFO] Robot action is unavailable (Stretch).")

@tool
def Dance():
    """ダンス"""
    if action:
        action.Dance()
    else:
        print("[INFO] Robot action is unavailable (Dance).")

@tool
def FrontJump():
    """前方にジャンプ"""
    if action:
        action.FrontJunmp()
    else:
        print("[INFO] Robot action is unavailable (FrontJump).")

@tool
def Heart():
    """ハートを描く"""
    if action:
        action.Heart()
    else:
        print("[INFO] Robot action is unavailable (Heart).")

@tool
def FrontFlip():
    """バク転"""
    if action:
        action.FrontFlip()
    else:
        print("[INFO] Robot action is unavailable (FrontFlip).")


tools = [StandUp, SitDown, Stretch, Dance, FrontJump, Heart, FrontFlip]

# -----------------------------
#  2) LLM (GPT) 関連セットアップ
# -----------------------------
model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def create_tool_agent(model, tools):
    def should_continue(state: MessagesState) -> Literal["tools", END]:
        messages = state['messages']
        last_message = messages[-1]
        if last_message.tool_calls:
            return "tools"
        return END

    def call_model(state: MessagesState):
        messages = state['messages']
        response = model.bind_tools(tools).invoke(messages)
        return {"messages": [response]}

    workflow = StateGraph(MessagesState)
    tool_node = ToolNode(tools)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)

    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", should_continue)
    workflow.add_edge("tools", 'agent')

    checkpointer = MemorySaver()
    app = workflow.compile(checkpointer=checkpointer)
    return app

app = create_tool_agent(model, tools)

# -----------------------------
#  3) OpenAI 非同期クライアント
# -----------------------------
async_openai_client = AsyncOpenAI()

# -----------------------------
#  4) MP3バイナリを直接再生する関数
# -----------------------------
def play_mp3_bytes(mp3_data: bytes):
    global is_tts_playing
    is_tts_playing = True
    try:
        audio_stream = io.BytesIO(mp3_data)
        audio_segment = AudioSegment.from_file(audio_stream, format="mp3")
        play(audio_segment)
    finally:
        is_tts_playing = False

# -----------------------------
#  5) 音声 → テキスト (Whisper)
# -----------------------------
async def stream_audio_to_text(
    audio_chunk: bytes,
    samplerate: int = 16000,
    num_channels: int = 1,
    sampwidth: int = 2
) -> str:
    """
    生PCMデータをWAV化し、一時ファイルとして保存。
    language="ja" を指定して日本語を優先認識する。
    """
    global current_timing
    current_timing["transcription_start"] = time.time()  # 計測開始

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wf:
        wf.setnchannels(num_channels)
        wf.setsampwidth(sampwidth)  # 16bit
        wf.setframerate(samplerate)
        wf.writeframesraw(audio_chunk)
    wav_buffer.seek(0)

    try:
        with NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
            tmpfile.write(wav_buffer.getvalue())
            tmpfile.flush()
            tmpfilename = tmpfile.name

        with open(tmpfilename, "rb") as f:
            response = await async_openai_client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                language="ja"  # 日本語
            )
        os.remove(tmpfilename)

        transcription = response.text.strip()
    except Exception as e:
        print(f"[ERROR in Whisper] {e}")
        transcription = ""

    current_timing["transcription_end"] = time.time()  # 計測終了
    return transcription

# -----------------------------
#  6) テキスト → 音声 (TTS)
# -----------------------------
async def text_to_speech(text: str) -> bytes:
    """
    GPTの返答をOpenAI TTSにかけて、音声(MP3バイナリ)として返す。
    """
    global current_timing
    current_timing["TTS_start"] = time.time()  # 計測開始

    mp3_data = b""
    try:
        response = await async_openai_client.audio.speech.create(
            model="tts-1",
            input=text,
            voice="alloy"
        )
        mp3_data = response.content
    except Exception as e:
        print(f"[ERROR in TTS] {e}")

    current_timing["TTS_end"] = time.time()  # 計測終了
    return mp3_data

# -----------------------------
#  7) 音声入力コールバック
# -----------------------------
def audio_callback(indata, frames, time_info, status, audio_queue: asyncio.Queue):
    """
    マイクからの音声を float32 -> int16 に変換しつつ
    無音を判定して、「話し終わり」でaudio_queueへ一括送信する。
    """
    if status:
        print(f"[Status] {status}")

    global is_tts_playing
    if is_tts_playing:
        # TTS再生中はユーザーの音声を無視
        return

    global current_buffer, silence_count, current_timing

    # float32 -> int16
    audio_data = np.int16(indata * 32767).tobytes()
    
    # 音量チェック(単純に絶対値の平均で判定)
    amplitude = np.mean(np.abs(indata) * 32767)
    is_silent = (amplitude < SILENCE_THRESHOLD)

    # 「最初に音声を検知した瞬間」をスタートとして記録
    if not is_silent and current_timing["capture_start"] is None:
        current_timing["capture_start"] = time.time()

    if not is_silent:
        # 喋っている: buffer に追加し、silence_countをリセット
        current_buffer.extend(audio_data)
        silence_count = 0
    else:
        # 無音
        silence_count += 1
        if silence_count >= MAX_SILENCE_BLOCKS and len(current_buffer) > 0:
            # 一定数以上 無音が続いたら、ユーザー発話終了と判断
            # capture_end を記録
            current_timing["capture_end"] = time.time()

            try:
                audio_queue.put_nowait(bytes(current_buffer))
            except asyncio.QueueFull:
                pass

            current_buffer = bytearray()
            silence_count = 0
        else:
            pass

# -----------------------------
#  8) 音声 → Whisper → GPT → TTS → mp3再生
# -----------------------------
async def run_transcription_loop(audio_queue: asyncio.Queue):
    """
    キューに入った「1ユーザー発話分の音声」を処理して応答する。
    """
    while True:
        audio_chunk = await audio_queue.get()
        print("[INFO] ユーザーの発話を検知。処理を開始します。")

        # Whisper
        transcription = await stream_audio_to_text(audio_chunk)
        if not transcription:
            reset_timing_info()
            continue
        print(f"[Transcription] {transcription}")

        # LLM解析
        current_timing["LLM_start"] = time.time()
        final_state = app.invoke(
            {"messages": [HumanMessage(content=transcription)]},
            config={"configurable": {"thread_id": 42}}
        )
        current_timing["LLM_end"] = time.time()

        response_text = final_state["messages"][-1].content
        print(f"[GPT Response] {response_text}")

        # TTS
        tts_audio = await text_to_speech(response_text)
        if tts_audio:
            with open("response_audio.mp3", "wb") as f:
                f.write(tts_audio)
            print("[Audio Saved] response_audio.mp3")
            play_mp3_bytes(tts_audio)
        else:
            print("[Audio Not Generated]")

        # 処理結果(音声取得～ロボット行動含む)を出力
        print_timing_info()
        reset_timing_info()

# -----------------------------
#  9) メイン関数
# -----------------------------
async def main():
    audio_queue = asyncio.Queue(maxsize=5)
    samplerate = 16000
    block_time = 1.0

    def sd_callback(indata, frames, time_info, status):
        audio_callback(indata, frames, time_info, status, audio_queue)

    transcription_task = asyncio.create_task(run_transcription_loop(audio_queue))

    # 録音開始
    print("=== 録音を開始します。ユーザが話し終わると応答を出力します。 ===")
    with sd.InputStream(
        channels=1,
        samplerate=samplerate,
        blocksize=int(samplerate * block_time),
        callback=sd_callback,
    ):
        try:
            while True:
                await asyncio.sleep(1.0)
        except KeyboardInterrupt:
            print("=== Ctrl+C で終了します ===")

    transcription_task.cancel()
    try:
        await transcription_task
    except asyncio.CancelledError:
        pass

# -----------------------------
# 10) スクリプトエントリポイント
# -----------------------------
if __name__ == "__main__":
    asyncio.run(main())
