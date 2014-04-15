/**
 * 
 */
package simulator;

import simulator.gpu.RuntimeFrameworkInterface;
import simulator.method.SimulationMethod;

/**
 * @author mcopik
 *
 */
public class SimulationSettings
{
	private SimulationMethod simMethod;

	private RuntimeFrameworkInterface simulator = null;
	private boolean simulatorTest = false;

	public SimulationSettings(SimulationMethod simMethod)
	{
		this.simMethod = simMethod;
	}

	public SimulationSettings(SimulationMethod simMethod, RuntimeFrameworkInterface simulator)
	{
		this.simMethod = simMethod;
		this.simulator = simulator;
	}

	public void runSimulatorTest()
	{
		simulatorTest = true;
	}

	public boolean isSimulatorTest()
	{
		return simulatorTest;
	}

	public void setMethod(SimulationMethod simMethod)
	{
		this.simMethod = simMethod;
	}

	public SimulationMethod getMethod()
	{
		return simMethod;
	}

	public boolean selectedStandardEngine()
	{
		return simulator == null;
	}

	public RuntimeFrameworkInterface getSimulatorFramework()
	{
		return simulator;
	}
}
