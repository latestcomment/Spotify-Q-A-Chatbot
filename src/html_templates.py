custom_css = '''
<style>
.chat-message {
    padding: 1.5rem; 
    border-radius: 0.5rem; 
    margin-bottom: 1rem; 
    display: flex;
    width: 100%;  /* Ensure the chat message takes full width of the container */
}
.chat-message.user {
    background-color: #007acc;
    color: white;  /* User message text in white for readability */
}
.chat-message.bot {
    background-color: #f4f6f9;
    color: black;
}
.chat-message .avatar {
  width: 10%;  /* Shrink avatar width */
  display: flex;
  justify-content: center;  /* Center the avatar image */
  align-items: center;
}
.chat-message .avatar img {
  max-width: 60px;  /* Adjust avatar size */
  max-height: 60px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 90%;  /* Increase message area to take more space */
  padding: 0 1.5rem;
  word-wrap: break-word;  /* Ensure long text wraps nicely */
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://i.pinimg.com/564x/dc/43/68/dc4368160e34d488a8c3a6c703ec017d.jpg" style="max-height: 78px; max-width: 100px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://i.pinimg.com/564x/18/af/ff/18afffb4b76ac8738f8b36952f89330b.jpg" style="max-height: 78px; max-width: 100px; border-radius: 50%; object-fit: cover;">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''