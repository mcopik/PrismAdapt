package simulator.gpu.opencl.kernel.memory;

import simulator.gpu.opencl.kernel.expression.Expression;

public class ExpressionValue implements CLValue
{
	private Expression expression;

	public ExpressionValue(Expression expr)
	{
		this.expression = expr;
	}

	@Override
	public boolean validateAssignmentTo(VariableInterface type)
	{
		return true;
	}

	@Override
	public Expression getSource()
	{
		return expression;
	}
}
