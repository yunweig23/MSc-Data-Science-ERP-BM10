using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using CesiumForUnity;
using Unity.Mathematics;
using UnityEngine;
using UnityEngine.Networking;

namespace NorthSeaEmissions
{
    [Serializable]
    public sealed class PortRecord
    {
        public string name;
        public double latitude;
        public double longitude;
    }

    [Serializable]
    public sealed class CorridorSummary
    {
        public int voyageCount;
        public double totalCo2Kg;
        public double weightedCo2PerKm;
        public double directionalBalancePercent;
        public int originToDestinationVoyages;
        public int destinationToOriginVoyages;
        public double roRoCo2SharePercent;
        public double passengerCo2SharePercent;
    }

    [Serializable]
    public sealed class AnnualEvidence
    {
        public int year;
        public int voyageCount;
        public int vesselCount;
        public double totalCo2Kg;
        public double weightedCo2PerKm;
        public int originToDestinationVoyages;
        public int destinationToOriginVoyages;
        public double roRoCo2SharePercent;
        public double passengerCo2SharePercent;
        public double portEnergyCo2SharePercent;
    }

    [Serializable]
    public sealed class CorridorDisplay
    {
        public double routeHeightMeters = 2500.0;
        public float animationSeconds = 12.0f;
        public float yearDisplaySeconds = 5.0f;
        public int routeSamples = 72;
    }

    [Serializable]
    public sealed class CorridorRecord
    {
        public string corridorId;
        public string title;
        public string analysisWindow;
        public PortRecord origin;
        public PortRecord destination;
        public CorridorSummary summary;
        public AnnualEvidence[] annualEvidence;
        public CorridorDisplay display;
    }

    public sealed class CorridorDemoController : MonoBehaviour
    {
        private const string DataFileName = "harwich_rotterdam_corridor.json";
        private const int MaxMarkersPerDirection = 8;
        private const float VoyagesPerMarker = 80.0f;
        private const float ShorePowerEfficiency = 0.70f;

        private static readonly Color DeepGreen = new Color(0.055f, 0.27f, 0.20f, 1.0f);
        private static readonly Color MidGreen = new Color(0.20f, 0.43f, 0.34f, 1.0f);
        private static readonly Color LowIntensityGreen = new Color(0.42f, 0.56f, 0.32f, 1.0f);
        private static readonly Color PassengerYellow = new Color(0.98f, 0.78f, 0.14f, 1.0f);
        private static readonly Color EmissionOrange = new Color(0.90f, 0.38f, 0.14f, 1.0f);
        private static readonly Color RouteLowBlue = new Color(0.20f, 0.66f, 0.82f, 1.0f);
        private static readonly Color RouteHighViolet = new Color(0.49f, 0.29f, 0.72f, 1.0f);
        private static readonly Color WarmWhite = new Color(0.975f, 0.98f, 0.96f, 0.95f);
        private static readonly Color MutedText = new Color(0.34f, 0.40f, 0.36f, 1.0f);
        private static readonly Color TrackColor = new Color(0.84f, 0.87f, 0.82f, 1.0f);

        private CesiumGeoreference _georeference;
        private CorridorRecord _corridor;
        private AnnualEvidence _currentEvidence;
        private Transform _visualRoot;
        private LineRenderer _routeLine;
        private Material _routeMaterial;
        private Material _roRoMaterial;
        private Material _passengerMaterial;
        private readonly List<FlowMarker> _eastboundMarkers = new List<FlowMarker>();
        private readonly List<FlowMarker> _westboundMarkers = new List<FlowMarker>();
        private Transform _originMarker;
        private Transform _destinationMarker;
        private Vector3[] _routePositions;
        private float _flowProgress;
        private float _yearClock;
        private int _yearIndex;
        private bool _timelinePlaying = true;
        private bool _ready;
        private string _status = "Loading corridor evidence...";

        private float _roRoReduction;
        private float _passengerReduction;
        private float _shorePowerCoverage;
        private double _scenarioTotalCo2Kg;
        private double _scenarioAvoidedCo2Kg;
        private double _scenarioReductionPercent;

        private GUIStyle _panelStyle;
        private GUIStyle _titleStyle;
        private GUIStyle _sectionStyle;
        private GUIStyle _labelStyle;
        private GUIStyle _valueStyle;
        private GUIStyle _smallValueStyle;
        private GUIStyle _noteStyle;
        private GUIStyle _buttonStyle;
        private GUIStyle _yearButtonStyle;
        private GUIStyle _activeYearButtonStyle;
        private GUIStyle _portTagStyle;
        private Texture2D _panelTexture;
        private Texture2D _buttonTexture;
        private Texture2D _inactiveButtonTexture;
        private Texture2D _activeYearTexture;
        private Texture2D _portTagTexture;
        private Texture2D _greenTexture;
        private Texture2D _tealTexture;
        private Texture2D _orangeTexture;
        private Texture2D _trackTexture;

        private sealed class FlowMarker
        {
            public Transform transform;
            public Renderer[] renderers;
            public float offset;
            public bool reverse;
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.AfterSceneLoad)]
        private static void Bootstrap()
        {
            EnsureInstance();
        }

        public static void EnsureInstance()
        {
            if (FindFirstObjectByType<CorridorDemoController>() != null)
            {
                return;
            }

            GameObject controllerObject = new GameObject("HarwichRotterdamCorridorDemo");
            DontDestroyOnLoad(controllerObject);
            controllerObject.AddComponent<CorridorDemoController>();
            Debug.Log("Created Harwich-Rotterdam corridor controller.");
        }

        private IEnumerator Start()
        {
            _georeference = FindFirstObjectByType<CesiumGeoreference>();
            if (_georeference == null)
            {
                _status = "CesiumGeoreference was not found in the active scene.";
                yield break;
            }

            ConfigureCamera();
            yield return LoadCorridorData();

            if (_corridor == null || _corridor.annualEvidence == null || _corridor.annualEvidence.Length == 0)
            {
                _status = "Annual corridor evidence was not available.";
                yield break;
            }

            BuildVisualization();
            SetYear(0, false);
            _ready = true;
            Debug.Log("Corridor demo ready with " + _corridor.annualEvidence.Length + " annual evidence records.");
#if UNITY_EDITOR
            StartCoroutine(CaptureEditorPreview());
#endif
        }

#if UNITY_EDITOR
        private IEnumerator CaptureEditorPreview()
        {
            yield return new WaitForSeconds(6.0f);
            string previewPath = Path.Combine(Path.GetTempPath(), "north_sea_corridor_preview.png");
            ScreenCapture.CaptureScreenshot(previewPath);
            Debug.Log("Corridor preview saved to " + previewPath);
        }
#endif

        private void Update()
        {
            if (!_ready || _currentEvidence == null)
            {
                return;
            }

            if (_corridor == null || _corridor.annualEvidence == null || _corridor.annualEvidence.Length == 0)
            {
                _ready = false;
                _status = "Restart Play mode to load the updated annual evidence.";
                return;
            }

            if (_timelinePlaying)
            {
                float duration = Mathf.Max(2.0f, _corridor.display.animationSeconds);
                _flowProgress = Mathf.Repeat(_flowProgress + Time.deltaTime / duration, 1.0f);
                _yearClock += Time.deltaTime;

                if (_yearClock >= Mathf.Max(2.0f, _corridor.display.yearDisplaySeconds))
                {
                    _yearClock = 0.0f;
                    SetYear((_yearIndex + 1) % _corridor.annualEvidence.Length, false);
                }
            }

            UpdateMarkers(_eastboundMarkers);
            UpdateMarkers(_westboundMarkers);
        }

        private void UpdateMarkers(List<FlowMarker> markers)
        {
            foreach (FlowMarker marker in markers)
            {
                if (!marker.transform.gameObject.activeSelf)
                {
                    continue;
                }

                float markerProgress = Mathf.Repeat(_flowProgress + marker.offset, 1.0f);
                if (marker.reverse)
                {
                    markerProgress = 1.0f - markerProgress;
                }

                UpdateVessel(marker.transform, markerProgress);
            }
        }

        private IEnumerator LoadCorridorData()
        {
            string path = Path.Combine(Application.streamingAssetsPath, DataFileName);
            string json;

            if (path.Contains("://"))
            {
                using UnityWebRequest request = UnityWebRequest.Get(path);
                yield return request.SendWebRequest();
                if (request.result != UnityWebRequest.Result.Success)
                {
                    _status = "Could not load corridor data: " + request.error;
                    yield break;
                }

                json = request.downloadHandler.text;
            }
            else
            {
                if (!File.Exists(path))
                {
                    _status = "Corridor data file was not found: " + path;
                    yield break;
                }

                json = File.ReadAllText(path);
            }

            _corridor = JsonUtility.FromJson<CorridorRecord>(json);
            if (_corridor == null || _corridor.origin == null || _corridor.destination == null)
            {
                _status = "Corridor data could not be parsed.";
            }
        }

        private void BuildVisualization()
        {
            GameObject rootObject = new GameObject("CorridorVisuals");
            rootObject.transform.SetParent(_georeference.transform, false);
            _visualRoot = rootObject.transform;

            int sampleCount = Mathf.Clamp(_corridor.display.routeSamples, 24, 160);
            _routePositions = new Vector3[sampleCount];

            for (int i = 0; i < sampleCount; i++)
            {
                double t = i / (double)(sampleCount - 1);
                double longitude = Lerp(_corridor.origin.longitude, _corridor.destination.longitude, t);
                double latitude = Lerp(_corridor.origin.latitude, _corridor.destination.latitude, t);
                // The raised ribbon is a symbolic corridor encoding, not a physical altitude.
                double arcLift = Math.Sin(Math.PI * t) * 12000.0;
                _routePositions[i] = ToUnityPosition(longitude, latitude, _corridor.display.routeHeightMeters + arcLift);
            }

            _roRoMaterial = CreateMaterial(EmissionOrange);
            _passengerMaterial = CreateMaterial(PassengerYellow);
            CreateRouteLine();
            _originMarker = CreatePortMarker(_corridor.origin);
            _destinationMarker = CreatePortMarker(_corridor.destination);
            CreateFlowMarkers();
        }

        private void CreateRouteLine()
        {
            GameObject routeObject = new GameObject("HistoricalCorridorRibbon");
            routeObject.transform.SetParent(_visualRoot, false);

            _routeLine = routeObject.AddComponent<LineRenderer>();
            _routeLine.useWorldSpace = false;
            _routeLine.positionCount = _routePositions.Length;
            _routeLine.SetPositions(_routePositions);
            _routeLine.numCornerVertices = 8;
            _routeLine.numCapVertices = 10;
            _routeMaterial = CreateMaterial(RouteLowBlue);
            _routeLine.material = _routeMaterial;
        }

        private Transform CreatePortMarker(PortRecord port)
        {
            GameObject marker = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
            marker.name = port.name;
            marker.transform.SetParent(_visualRoot, false);
            marker.transform.localPosition = ToUnityPosition(
                port.longitude,
                port.latitude,
                _corridor.display.routeHeightMeters + 5000.0);
            marker.transform.localScale = new Vector3(2400.0f, 5000.0f, 2400.0f);
            marker.GetComponent<Renderer>().material = CreateMaterial(DeepGreen);
            RemoveCollider(marker);

            GameObject cap = GameObject.CreatePrimitive(PrimitiveType.Sphere);
            cap.name = port.name + " beacon";
            cap.transform.SetParent(marker.transform, false);
            cap.transform.localPosition = new Vector3(0.0f, 1.08f, 0.0f);
            cap.transform.localScale = new Vector3(1.35f, 0.18f, 1.35f);
            cap.GetComponent<Renderer>().material = CreateMaterial(EmissionOrange);
            RemoveCollider(cap);

            return marker.transform;
        }

        private void CreateFlowMarkers()
        {
            for (int i = 0; i < MaxMarkersPerDirection; i++)
            {
                _eastboundMarkers.Add(CreateVessel(
                    "Harwich to Rotterdam flow " + (i + 1),
                    i / (float)MaxMarkersPerDirection,
                    false));
                _westboundMarkers.Add(CreateVessel(
                    "Rotterdam to Harwich flow " + (i + 1),
                    (i + 0.5f) / MaxMarkersPerDirection,
                    true));
            }
        }

        private FlowMarker CreateVessel(string objectName, float offset, bool reverse)
        {
            GameObject root = new GameObject(objectName);
            root.transform.SetParent(_visualRoot, false);

            GameObject hull = GameObject.CreatePrimitive(PrimitiveType.Cube);
            hull.name = "Hull";
            hull.transform.SetParent(root.transform, false);
            hull.transform.localScale = new Vector3(2000.0f, 900.0f, 5000.0f);
            Renderer hullRenderer = hull.GetComponent<Renderer>();
            hullRenderer.material = _roRoMaterial;
            RemoveCollider(hull);

            GameObject bridge = GameObject.CreatePrimitive(PrimitiveType.Cube);
            bridge.name = "Bridge";
            bridge.transform.SetParent(root.transform, false);
            bridge.transform.localPosition = new Vector3(0.0f, 900.0f, -650.0f);
            bridge.transform.localScale = new Vector3(1350.0f, 900.0f, 1700.0f);
            Renderer bridgeRenderer = bridge.GetComponent<Renderer>();
            bridgeRenderer.material = _roRoMaterial;
            RemoveCollider(bridge);

            return new FlowMarker
            {
                transform = root.transform,
                renderers = new[] { hullRenderer, bridgeRenderer },
                offset = offset,
                reverse = reverse
            };
        }

        private static void RemoveCollider(GameObject target)
        {
            Collider collider = target.GetComponent<Collider>();
            if (collider != null)
            {
                Destroy(collider);
            }
        }

        private void SetYear(int index, bool pauseTimeline)
        {
            _yearIndex = Mathf.Clamp(index, 0, _corridor.annualEvidence.Length - 1);
            _currentEvidence = _corridor.annualEvidence[_yearIndex];
            _yearClock = 0.0f;

            if (pauseTimeline)
            {
                _timelinePlaying = false;
            }

            _status = _currentEvidence.year + " historical corridor evidence";
            ApplyEvidenceToScene();
        }

        private void ApplyEvidenceToScene()
        {
            if (_currentEvidence == null || _routeLine == null)
            {
                return;
            }

            RecalculateScenario();
            UpdateRouteAppearance();
            ConfigureDirectionMarkers(_eastboundMarkers, _currentEvidence.originToDestinationVoyages);
            ConfigureDirectionMarkers(_westboundMarkers, _currentEvidence.destinationToOriginVoyages);
        }

        private void RecalculateScenario()
        {
            double roRoReduction = (_currentEvidence.roRoCo2SharePercent / 100.0) * _roRoReduction;
            double passengerReduction = (_currentEvidence.passengerCo2SharePercent / 100.0) * _passengerReduction;
            double shipTypeReduction = Math.Min(1, roRoReduction + passengerReduction);
            double remainingAfterShipMeasures = 1.0 - shipTypeReduction;
            double shorePowerReduction = remainingAfterShipMeasures
                * (_currentEvidence.portEnergyCo2SharePercent / 100.0)
                * _shorePowerCoverage
                * ShorePowerEfficiency;
            double reductionFraction = Math.Min(1, shipTypeReduction + shorePowerReduction);

            _scenarioAvoidedCo2Kg = _currentEvidence.totalCo2Kg * reductionFraction;
            _scenarioTotalCo2Kg = _currentEvidence.totalCo2Kg - _scenarioAvoidedCo2Kg;
            _scenarioReductionPercent = reductionFraction * 100.0;
        }

        private void UpdateRouteAppearance()
        {
            double minimumCo2 = double.MaxValue;
            double maximumCo2 = double.MinValue;
            double minimumIntensity = double.MaxValue;
            double maximumIntensity = double.MinValue;

            foreach (AnnualEvidence evidence in _corridor.annualEvidence)
            {
                minimumCo2 = Math.Min(minimumCo2, evidence.totalCo2Kg);
                maximumCo2 = Math.Max(maximumCo2, evidence.totalCo2Kg);
                minimumIntensity = Math.Min(minimumIntensity, evidence.weightedCo2PerKm);
                maximumIntensity = Math.Max(maximumIntensity, evidence.weightedCo2PerKm);
            }

            float totalScale = Mathf.InverseLerp((float)minimumCo2, (float)maximumCo2, (float)_scenarioTotalCo2Kg);
            float scenarioIntensity = (float)(_currentEvidence.weightedCo2PerKm
                * (_scenarioTotalCo2Kg / _currentEvidence.totalCo2Kg));
            float intensityScale = Mathf.InverseLerp((float)minimumIntensity, (float)maximumIntensity, scenarioIntensity);
            float width = Mathf.Lerp(2400.0f, 6200.0f, Mathf.Sqrt(Mathf.Clamp01(totalScale)));
            Color routeColor = Color.Lerp(RouteLowBlue, RouteHighViolet, Mathf.Clamp01(intensityScale));

            _routeLine.startWidth = width;
            _routeLine.endWidth = width;
            _routeLine.startColor = new Color(routeColor.r, routeColor.g, routeColor.b, 0.95f);
            _routeLine.endColor = new Color(routeColor.r, routeColor.g, routeColor.b, 0.72f);
            _routeMaterial.color = routeColor;
        }

        private void ConfigureDirectionMarkers(List<FlowMarker> markers, int voyageCount)
        {
            int activeCount = Mathf.Clamp(Mathf.CeilToInt(voyageCount / VoyagesPerMarker), 1, markers.Count);
            float roRoFraction = (float)(_currentEvidence.roRoCo2SharePercent
                / Math.Max(1.0, _currentEvidence.roRoCo2SharePercent + _currentEvidence.passengerCo2SharePercent));
            int roRoCount = Mathf.Clamp(Mathf.RoundToInt(activeCount * roRoFraction), 0, activeCount);

            for (int i = 0; i < markers.Count; i++)
            {
                bool active = i < activeCount;
                markers[i].transform.gameObject.SetActive(active);
                if (!active)
                {
                    continue;
                }

                Material material = i < roRoCount ? _roRoMaterial : _passengerMaterial;
                foreach (Renderer renderer in markers[i].renderers)
                {
                    renderer.material = material;
                }
            }
        }

        private void UpdateVessel(Transform vessel, float progress)
        {
            if (vessel == null || _routePositions == null || _routePositions.Length < 2)
            {
                return;
            }

            float scaled = Mathf.Clamp01(progress) * (_routePositions.Length - 1);
            int index = Mathf.Min(Mathf.FloorToInt(scaled), _routePositions.Length - 2);
            float segmentProgress = scaled - index;
            Vector3 position = Vector3.Lerp(_routePositions[index], _routePositions[index + 1], segmentProgress);
            Vector3 direction = (_routePositions[index + 1] - _routePositions[index]).normalized;

            vessel.localPosition = position + Vector3.up * 1450.0f;
            if (direction.sqrMagnitude > 0.001f)
            {
                vessel.localRotation = Quaternion.LookRotation(direction, Vector3.up);
            }
        }

        private void ConfigureCamera()
        {
            Camera activeCamera = Camera.main;
            if (activeCamera == null)
            {
                return;
            }

            Transform cameraTransform = activeCamera.transform;
            cameraTransform.SetParent(_georeference.transform, false);
            cameraTransform.localPosition = new Vector3(0.0f, 50000.0f, -150000.0f);
            cameraTransform.localRotation = Quaternion.Euler(49.0f, 0.0f, 0.0f);
            activeCamera.fieldOfView = 42.0f;
        }

        private Vector3 ToUnityPosition(double longitude, double latitude, double height)
        {
            double3 ecef = _georeference.ellipsoid.LongitudeLatitudeHeightToCenteredFixed(
                new double3(longitude, latitude, height));
            double3 unityPosition = _georeference.TransformEarthCenteredEarthFixedPositionToUnity(ecef);
            return new Vector3((float)unityPosition.x, (float)unityPosition.y, (float)unityPosition.z);
        }

        private static double Lerp(double start, double end, double t)
        {
            return start + (end - start) * t;
        }

        private static Material CreateMaterial(Color color)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Unlit");
            if (shader == null)
            {
                shader = Shader.Find("Sprites/Default");
            }

            Material material = new Material(shader);
            material.color = color;
            return material;
        }

        private void EnsureGuiStyles()
        {
            if (_panelStyle != null && _sectionStyle != null && _greenTexture != null)
            {
                return;
            }

            _panelTexture = MakeTexture(WarmWhite);
            _buttonTexture = MakeTexture(DeepGreen);
            _inactiveButtonTexture = MakeTexture(new Color(0.90f, 0.92f, 0.88f, 1.0f));
            _activeYearTexture = MakeTexture(MidGreen);
            _portTagTexture = MakeTexture(new Color(DeepGreen.r, DeepGreen.g, DeepGreen.b, 0.92f));
            _greenTexture = MakeTexture(MidGreen);
            _tealTexture = MakeTexture(PassengerYellow);
            _orangeTexture = MakeTexture(EmissionOrange);
            _trackTexture = MakeTexture(TrackColor);

            _panelStyle = new GUIStyle(GUI.skin.box)
            {
                normal = { background = _panelTexture },
                padding = new RectOffset(22, 22, 18, 18)
            };
            _titleStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 24,
                fontStyle = FontStyle.Bold,
                normal = { textColor = DeepGreen }
            };
            _sectionStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 15,
                fontStyle = FontStyle.Bold,
                normal = { textColor = DeepGreen }
            };
            _labelStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 12,
                normal = { textColor = MutedText }
            };
            _valueStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 18,
                fontStyle = FontStyle.Bold,
                normal = { textColor = DeepGreen }
            };
            _smallValueStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 14,
                fontStyle = FontStyle.Bold,
                normal = { textColor = DeepGreen }
            };
            _noteStyle = new GUIStyle(GUI.skin.label)
            {
                fontSize = 11,
                wordWrap = true,
                normal = { textColor = MutedText }
            };
            _buttonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 12,
                fontStyle = FontStyle.Bold,
                normal = { background = _buttonTexture, textColor = Color.white },
                hover = { background = _buttonTexture, textColor = Color.white },
                active = { background = _buttonTexture, textColor = Color.white }
            };
            _yearButtonStyle = new GUIStyle(GUI.skin.button)
            {
                fontSize = 12,
                fontStyle = FontStyle.Bold,
                normal = { background = _inactiveButtonTexture, textColor = DeepGreen },
                hover = { background = _inactiveButtonTexture, textColor = DeepGreen }
            };
            _activeYearButtonStyle = new GUIStyle(_yearButtonStyle)
            {
                normal = { background = _activeYearTexture, textColor = Color.white },
                hover = { background = _activeYearTexture, textColor = Color.white }
            };
            _portTagStyle = new GUIStyle(GUI.skin.box)
            {
                fontSize = 12,
                fontStyle = FontStyle.Bold,
                alignment = TextAnchor.MiddleCenter,
                normal = { background = _portTagTexture, textColor = Color.white },
                padding = new RectOffset(10, 10, 6, 6)
            };
        }

        private void OnGUI()
        {
            EnsureGuiStyles();
            float scale = Mathf.Clamp(Screen.width / 1920.0f, 0.72f, 1.0f);
            Matrix4x4 previousMatrix = GUI.matrix;
            GUI.matrix = Matrix4x4.TRS(Vector3.zero, Quaternion.identity, new Vector3(scale, scale, 1.0f));
            float viewWidth = Screen.width / scale;

            DrawEvidencePanel();
            if (_ready)
            {
                DrawScenarioPanel(viewWidth);
            }

            GUI.matrix = previousMatrix;
            DrawPortTag(_originMarker, "HARWICH");
            DrawPortTag(_destinationMarker, "ROTTERDAM");
        }

        private void DrawEvidencePanel()
        {
            Rect panelRect = new Rect(28.0f, 28.0f, 470.0f, _ready ? 430.0f : 105.0f);
            GUI.Box(panelRect, GUIContent.none, _panelStyle);
            GUI.Label(new Rect(50.0f, 46.0f, 410.0f, 35.0f), _ready ? _corridor.title : "North Sea corridor", _titleStyle);
            GUI.Label(new Rect(50.0f, 80.0f, 410.0f, 22.0f), _status, _labelStyle);

            if (!_ready)
            {
                return;
            }

            if (_corridor == null || _corridor.annualEvidence == null || _corridor.annualEvidence.Length == 0)
            {
                GUI.Label(new Rect(50.0f, 108.0f, 398.0f, 42.0f),
                    "Stop and restart Play mode to load the updated annual evidence.", _noteStyle);
                return;
            }

            GUI.Label(new Rect(50.0f, 108.0f, 150.0f, 24.0f), "YEAR TIMELINE", _sectionStyle);
            for (int i = 0; i < _corridor.annualEvidence.Length; i++)
            {
                GUIStyle style = i == _yearIndex ? _activeYearButtonStyle : _yearButtonStyle;
                if (GUI.Button(new Rect(50.0f + i * 72.0f, 136.0f, 64.0f, 30.0f),
                    _corridor.annualEvidence[i].year.ToString(), style))
                {
                    SetYear(i, true);
                }
            }

            if (GUI.Button(new Rect(350.0f, 136.0f, 98.0f, 30.0f),
                _timelinePlaying ? "Pause" : "Play", _buttonStyle))
            {
                _timelinePlaying = !_timelinePlaying;
                _yearClock = 0.0f;
            }

            DrawKpi(50.0f, 184.0f, "Voyages", _currentEvidence.voyageCount.ToString("N0"));
            DrawKpi(190.0f, 184.0f, "Total CO2", (_currentEvidence.totalCo2Kg / 1000000.0).ToString("N1") + "M kg");
            DrawKpi(335.0f, 184.0f, "CO2 / km", _currentEvidence.weightedCo2PerKm.ToString("N0") + " kg");

            GUI.Label(new Rect(50.0f, 244.0f, 180.0f, 22.0f), "DIRECTIONAL ACTIVITY", _sectionStyle);
            GUI.Label(new Rect(50.0f, 269.0f, 190.0f, 20.0f),
                "Harwich to Rotterdam  " + _currentEvidence.originToDestinationVoyages.ToString("N0"), _labelStyle);
            GUI.Label(new Rect(258.0f, 269.0f, 190.0f, 20.0f),
                "Rotterdam to Harwich  " + _currentEvidence.destinationToOriginVoyages.ToString("N0"), _labelStyle);
            DrawSplitBar(new Rect(50.0f, 291.0f, 398.0f, 10.0f),
                _currentEvidence.originToDestinationVoyages,
                _currentEvidence.destinationToOriginVoyages,
                _orangeTexture,
                _tealTexture);

            GUI.Label(new Rect(50.0f, 318.0f, 180.0f, 22.0f), "VISUAL ENCODING", _sectionStyle);
            DrawLegendItem(50.0f, 345.0f, _orangeTexture, "Ro-Ro vessel marker");
            DrawLegendItem(242.0f, 345.0f, _tealTexture, "Passenger vessel marker");
            GUI.Label(new Rect(50.0f, 371.0f, 398.0f, 20.0f),
                "Motion = direction  |  marker count = activity (about 80 voyages each)", _noteStyle);
            GUI.Label(new Rect(50.0f, 392.0f, 398.0f, 34.0f),
                "Wider ribbon = higher total CO2  |  blue to violet = lower to higher CO2 intensity", _noteStyle);
        }

        private void DrawScenarioPanel(float viewWidth)
        {
            float x = Mathf.Max(520.0f, viewWidth - 430.0f);
            Rect panelRect = new Rect(x, 28.0f, 402.0f, 430.0f);
            GUI.Box(panelRect, GUIContent.none, _panelStyle);
            GUI.Label(new Rect(x + 22.0f, 46.0f, 350.0f, 30.0f), "Decarbonisation scenario", _titleStyle);
            GUI.Label(new Rect(x + 22.0f, 80.0f, 350.0f, 34.0f),
                "Adjust illustrative measures for the selected year.", _noteStyle);

            float previousRoRo = _roRoReduction;
            float previousPassenger = _passengerReduction;
            float previousShore = _shorePowerCoverage;

            _roRoReduction = DrawScenarioSlider(x + 22.0f, 120.0f, "Ro-Ro CO2 reduction",
                _roRoReduction, 0.0f, 0.40f);
            _passengerReduction = DrawScenarioSlider(x + 22.0f, 178.0f, "Passenger CO2 reduction",
                _passengerReduction, 0.0f, 0.40f);
            _shorePowerCoverage = DrawScenarioSlider(x + 22.0f, 236.0f, "Shore-power coverage",
                _shorePowerCoverage, 0.0f, 1.0f);

            if (!Mathf.Approximately(previousRoRo, _roRoReduction)
                || !Mathf.Approximately(previousPassenger, _passengerReduction)
                || !Mathf.Approximately(previousShore, _shorePowerCoverage))
            {
                ApplyEvidenceToScene();
            }

            GUI.Label(new Rect(x + 22.0f, 300.0f, 170.0f, 20.0f), "Scenario CO2", _labelStyle);
            GUI.Label(new Rect(x + 22.0f, 321.0f, 170.0f, 28.0f),
                (_scenarioTotalCo2Kg / 1000000.0).ToString("N1") + "M kg", _valueStyle);
            GUI.Label(new Rect(x + 210.0f, 300.0f, 150.0f, 20.0f), "Potential reduction", _labelStyle);
            GUI.Label(new Rect(x + 210.0f, 321.0f, 150.0f, 28.0f),
                _scenarioReductionPercent.ToString("N1") + "%", _valueStyle);

            GUI.Label(new Rect(x + 22.0f, 354.0f, 255.0f, 42.0f),
                "Illustrative comparison, not a forecast. Shore power applies a 70% reduction assumption to the auxiliary and boiler CO2 share.", _noteStyle);
            if (GUI.Button(new Rect(x + 285.0f, 365.0f, 88.0f, 30.0f), "Reset", _buttonStyle))
            {
                _roRoReduction = 0.0f;
                _passengerReduction = 0.0f;
                _shorePowerCoverage = 0.0f;
                ApplyEvidenceToScene();
            }
        }

        private void DrawKpi(float x, float y, string label, string value)
        {
            GUI.Label(new Rect(x, y, 130.0f, 20.0f), label, _labelStyle);
            GUI.Label(new Rect(x, y + 20.0f, 138.0f, 28.0f), value, _valueStyle);
        }

        private float DrawScenarioSlider(float x, float y, string label, float value, float minimum, float maximum)
        {
            GUI.Label(new Rect(x, y, 230.0f, 20.0f), label, _labelStyle);
            GUI.Label(new Rect(x + 280.0f, y, 70.0f, 20.0f), (value * 100.0f).ToString("N0") + "%", _smallValueStyle);
            float rawValue = GUI.HorizontalSlider(
                new Rect(x, y + 26.0f, 330.0f, 18.0f), value, minimum, maximum);
            return Mathf.Clamp(Mathf.Round(rawValue * 100.0f) / 100.0f, minimum, maximum);
        }

        private void DrawSplitBar(Rect rect, float first, float second, Texture2D firstTexture, Texture2D secondTexture)
        {
            GUI.DrawTexture(rect, _trackTexture);
            float total = Mathf.Max(1.0f, first + second);
            float firstWidth = rect.width * first / total;
            GUI.DrawTexture(new Rect(rect.x, rect.y, firstWidth, rect.height), firstTexture);
            GUI.DrawTexture(new Rect(rect.x + firstWidth, rect.y, rect.width - firstWidth, rect.height), secondTexture);
        }

        private void DrawLegendItem(float x, float y, Texture2D texture, string label)
        {
            GUI.DrawTexture(new Rect(x, y + 3.0f, 14.0f, 14.0f), texture);
            GUI.Label(new Rect(x + 21.0f, y, 170.0f, 22.0f), label, _labelStyle);
        }

        private void DrawPortTag(Transform marker, string label)
        {
            Camera activeCamera = Camera.main;
            if (!_ready || marker == null || activeCamera == null)
            {
                return;
            }

            Vector3 screenPosition = activeCamera.WorldToScreenPoint(marker.position);
            if (screenPosition.z <= 0.0f)
            {
                return;
            }

            float width = 112.0f;
            Rect tagRect = new Rect(
                screenPosition.x - width * 0.5f,
                Screen.height - screenPosition.y - 50.0f,
                width,
                28.0f);
            GUI.Box(tagRect, label, _portTagStyle);
        }

        private static Texture2D MakeTexture(Color color)
        {
            Texture2D texture = new Texture2D(1, 1);
            texture.SetPixel(0, 0, color);
            texture.Apply();
            return texture;
        }
    }

}
