var userInput = document.getElementById('user_input');
var userEmotion = document.getElementById('user_emotion');
var i = document.getElementById('i');
var esTmp = document.getElementById('ES_tmp');
var es = document.getElementById('ES');
var chatbotResponse = document.getElementById('chatbot_response');
var chatbotEmotion = document.getElementById('chatbot_emotion');

function submit() {
  // action for the submit button
  if (userInput.value == '') {
    window.alert("You don't input anything!");
    return;
  }
  // call the predict function of the backend
  var send = {"userInput":userInput.value, "i":i.innerText, "es":es.innerText, "esTmp":esTmp.innerText}
  predictEmo(send);
}

function clearAll() {
  userInput.value = '';
  userEmotion.value = '';
  chatbotResponse.value = '';
  chatbotEmotion.value = '';
}

function predictEmo(send) {
  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify(send)
  })
    .then(resp => {
      if (resp.ok)
        resp.json().then(data => {
          displayResult(data);
        });
    })
    .catch(err => {
      console.log("An error occured", err.message);
      window.alert("Oops! Something went wrong.");
    });
}

function displayResult(data) {
  userEmotion.value = data.E;
  chatbotEmotion.value = data.chatbot_E;
  chatbotResponse.value = data.chatbot_response_sentence;
  esTmp.innerText = data.ES_tmp;
  i.innerText = data.i;
  if (parseInt(data.i-1) != 0 && parseInt(data.i-1) %5 == 0){
    alert('Chatbot Restore Emotion!')
  }
}