/**
 * 
 */
package simulator.gpu.opencl.kernel.memory;

import simulator.gpu.opencl.kernel.expression.Expression;

/**
 * @author mcopik
 *
 */
public class RValue implements CLValue
{
	private final Expression rValue;

	public RValue(Expression expr)
	{
		rValue = expr;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.memory.CLValue#validateAssignmentTo(simulator.gpu.opencl.kernel.memory.VariableInterface)
	 */
	@Override
	public boolean validateAssignmentTo(VariableInterface type)
	{
		return true;
	}

	/* (non-Javadoc)
	 * @see simulator.gpu.opencl.kernel.memory.CLValue#getSource()
	 */
	@Override
	public Expression getSource()
	{
		return rValue;
	}

}
