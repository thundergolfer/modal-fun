function Spinner({ config }) {
  const ref = React.useRef(null);

  React.useEffect(() => {
    const spinner = new Spin.Spinner({
      lines: 13,
      color: "#ffffff",
      ...config,
    });
    spinner.spin(ref.current);
    return () => spinner.stop();
  }, [ref]);

  return <span ref={ref} />;
}

function truncate(str, n) {
  return str.length > n ? str.slice(0, n - 1) + "…" : str;
}

function Podcast({ podcast }) {
    const [isSending, setIsSending] = React.useState(false)
    const sendRequest = React.useCallback(async () => {
      // don't send again while we are sending
      if (isSending) return;
      // update state
      setIsSending(true)
      // send the actual request
      console.log(`Transcribing ${podcast.title} ${podcast.id}`);
      const formData = new FormData();
      formData.append("podcast_name", podcast.title);
      formData.append("podcast_id", podcast.id);
  
      const resp = await fetch("/transcribe", {
        method: "POST",
        body: formData,
      });

      if (resp.status !== 200) {
        throw new Error("An error occurred: " + resp.status);
      }
      const body = await resp.json();
      console.log("Received response:");
      console.log(body);
      // once the request is sent, update state again
      setIsSending(false)
    }, [isSending]); // update the callback if the state changes

  return (
    <div className="max-w-sm rounded overflow-hidden shadow-lg">
      <div className="px-6 py-4">
        <div className="font-bold text-xl mb-2">{podcast.title}</div>
        <p className="text-gray-700 text-base">
          {truncate(podcast.description, 200)}
        </p>
        <button 
            className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded" 
            disabled={isSending} 
            onClick={sendRequest}
        >
            {isSending ? <Spinner config={{}} /> : "Transcribe!"}
        </button>
      </div>
    </div>
  );
}

function PodcastList({ podcasts }) {
  console.log("podcasts:");
  console.log(podcasts);
  const listItems = podcasts.map((pod) => (
    <li key={pod.id}>
      <Podcast podcast={pod} />
    </li>
  ));
  return <ul>{listItems}</ul>;
}

function Result({ callId, selectedFile }) {
  const [result, setResult] = React.useState();
  const [intervalId, setIntervalId] = React.useState();

  React.useEffect(() => {
    if (result) {
      clearInterval(intervalId);
      return;
    }

    const _intervalID = setInterval(async () => {
      const resp = await fetch(`/result/${callId}`);
      if (resp.status === 200) {
        setResult(await resp.json());
      }
    }, 100);

    setIntervalId(_intervalID);

    return () => clearInterval(intervalId);
  }, [result]);

  return (
    <div class="flex items-center content-center justify-center space-x-4 ">
      <img src={URL.createObjectURL(selectedFile)} class="h-[300px]" />
      {!result && <Spinner config={{}} />}
      {result && (
        <p class="w-[200px] p-4 bg-zinc-200 rounded-lg whitespace-pre-wrap text-xs font-mono">
          {JSON.stringify(result, undefined, 1)}
        </p>
      )}
    </div>
  );
}

function Form({ onSubmit }) {
  const [podcastName, setPodcastName] = React.useState("");
  const onChange = (event) => {
    setPodcastName(event.target.value);
  };

  const handleSubmit = async (event) => {
    console.log("Handling submit");
    await onSubmit(podcastName);
  };

  return (
    <form className="flex flex-col space-y-4 items-center">
      <div className="text-2xl font-semibold text-gray-700">
        Modal Podcast Transcriber
      </div>
      <label>
        <strong>Podcast:</strong>
        <input
          type="text"
          value={podcastName}
          onChange={onChange}
          className="text-sm text-gray-900 bg-gray-50 rounded-lg border border-gray-300 cursor-pointer"
        />
      </label>
      <div>
        <button
          type="button"
          onClick={handleSubmit}
          disabled={!podcastName}
          className="bg-indigo-400 disabled:bg-zinc-500 hover:bg-indigo-600 text-white font-bold py-2 px-4 rounded text-sm"
        >
          Search
        </button>
      </div>
    </form>
  );
}

function App() {
  const [callId, setCallId] = React.useState();
  const [podcasts, setPodcasts] = React.useState();

  const handleSubmission = async (podcastName) => {
    const formData = new FormData();
    formData.append("podcast", podcastName);
    console.log(podcastName);

    const resp = await fetch("/podcasts", {
      method: "POST",
      body: formData,
    });

    if (resp.status !== 200) {
      throw new Error("An error occurred: " + resp.status);
    }
    const body = await resp.json();
    setPodcasts(body);
    //   setCallId(body.call_id);
  };

  return (
    <div className="absolute inset-0 bg-gradient-to-r from-green-300 via-green-500 to-green-300">
      <div className="mx-auto max-w-md py-8">
        <main className="rounded-xl bg-white p-6">
          {!callId && <Form onSubmit={handleSubmission} />}
          {podcasts && <PodcastList podcasts={podcasts} />}
          {callId && <Result callId={callId} selectedFile={selectedFile} />}
        </main>
      </div>
    </div>
  );
}

const container = document.getElementById("react");
ReactDOM.createRoot(container).render(<App />);
