using NorthSeaEmissions;
using UnityEditor;

[InitializeOnLoad]
internal static class CorridorDemoEditorBootstrap
{
    static CorridorDemoEditorBootstrap()
    {
        EditorApplication.playModeStateChanged -= HandlePlayModeState;
        EditorApplication.playModeStateChanged += HandlePlayModeState;
        EditorApplication.update -= EnsureControllerWhilePlaying;
        EditorApplication.update += EnsureControllerWhilePlaying;
    }

    private static void HandlePlayModeState(PlayModeStateChange state)
    {
        if (state == PlayModeStateChange.EnteredPlayMode)
        {
            CorridorDemoController.EnsureInstance();
        }
    }

    private static void EnsureControllerWhilePlaying()
    {
        if (EditorApplication.isPlaying)
        {
            CorridorDemoController.EnsureInstance();
        }
    }
}
